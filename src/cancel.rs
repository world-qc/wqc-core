//! Stops a computation whose client has gone. A node that gives up on
//! `/compute` closes the connection, which drops the handler; without this,
//! core would keep computing and the node's next sub-task would wait behind
//! a result nobody reads.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::engine::EngineError;

thread_local! {
    static FLAG: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Message of the error [`check`] returns.
pub const CANCELLED: &str = "computation cancelled: the client has gone";

/// Held by the handler while the computation runs. Dropping it, as hyper does
/// when the client closes the connection, cancels the computation; once the
/// computation has finished, nothing reads the flag.
pub struct OnDrop(Arc<AtomicBool>);

impl OnDrop {
    pub fn new() -> (Self, Arc<AtomicBool>) {
        let flag = Arc::new(AtomicBool::new(false));
        (Self(flag.clone()), flag)
    }
}

impl Drop for OnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Runs `f` on this thread with `flag` as what [`check`] reads.
pub fn run<T>(flag: Arc<AtomicBool>, f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FLAG.with(|f| *f.borrow_mut() = None);
        }
    }
    FLAG.with(|f| *f.borrow_mut() = Some(flag));
    let _reset = Reset;
    f()
}

/// `Err` once the computation on this thread has been cancelled. Outside
/// [`run`] it never is.
pub fn check() -> Result<(), EngineError> {
    let cancelled = FLAG.with(|f| {
        f.borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    });
    if cancelled {
        Err(EngineError::ExecutionFailed(CANCELLED.into()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropping_the_guard_cancels_only_inside_run() {
        assert!(check().is_ok());
        let (guard, flag) = OnDrop::new();
        run(flag.clone(), || {
            assert!(check().is_ok());
            drop(guard);
            assert_eq!(
                check().unwrap_err().to_string(),
                EngineError::ExecutionFailed(CANCELLED.into()).to_string()
            );
        });
        // The flag stays set, but this thread no longer runs that computation.
        assert!(flag.load(Ordering::Relaxed));
        assert!(check().is_ok());
    }
}
