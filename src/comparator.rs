use crate::server::KnnServer;
use crate::setup_polymul_fft;
use dyn_stack::DynStack;
use std::cell::RefCell;
use std::cmp::{Ord, Ordering};
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use tfhe::shortint::prelude::*;

#[derive(Eq, Copy, Clone)]
pub struct ClearItem {
    pub value: u64,
    pub class: u64,
}

impl Ord for ClearItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialOrd for ClearItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

impl PartialEq for ClearItem {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl fmt::Debug for ClearItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.value, self.class)
    }
}

/// This is our comparator which is used in the Batcher odd-even network.
pub trait Comparator {
    type Item;
    type Aux; // auxiliary information, e.g., FFT context

    fn compare(&self, vs: &mut [Self::Item], i: usize, j: usize);
    fn swap(&self, vs: &mut [Self::Item], i: usize, j: usize);
    fn compare_count(&self) -> usize;
}

pub struct ClearCmp<T> {
    counter: Rc<RefCell<usize>>,
    item_type: PhantomData<T>,
}

impl<T: Ord + Clone> ClearCmp<T> {
    /// Create a plaintext vector that implements `Comparator`.
    pub fn new() -> Self {
        Self {
            counter: Rc::new(RefCell::new(0)),
            item_type: PhantomData,
        }
    }
}

impl<T: Ord + Clone> Default for ClearCmp<T> {
    fn default() -> Self {
        ClearCmp::new()
    }
}

impl<T: Ord + Clone> Comparator for ClearCmp<T> {
    type Item = T;
    type Aux = ();

    fn compare(&self, vs: &mut [Self::Item], i: usize, j: usize) {
        if vs[i] > vs[j] {
            vs.swap(i, j);
        }
        *self.counter.borrow_mut() += 1;
    }

    fn swap(&self, vs: &mut [Self::Item], i: usize, j: usize) {
        vs.swap(i, j);
    }

    fn compare_count(&self) -> usize {
        *self.counter.borrow()
    }
}

pub struct EncItem {
    pub value: Ciphertext,
    pub class: Ciphertext,
}

impl EncItem {
    pub fn new(value: Ciphertext, class: Ciphertext) -> Self {
        Self { value, class }
    }

    pub fn decrypt(&self, client_key: &ClientKey) -> (u64, u64) {
        (
            client_key.decrypt(&self.value),
            client_key.decrypt(&self.class),
        )
    }
}

pub struct EncCmp {
    params: Parameters,
    server: Rc<RefCell<KnnServer>>,
    counter: Rc<RefCell<usize>>,
}

impl EncCmp {
    /// Create an encrypted vector that implements `Comparator`.
    /// A reference to `KnnServer` is needed because it has the cryptography context.
    /// The output is boxed.
    pub fn new(params: Parameters, server: Rc<RefCell<KnnServer>>) -> Self {
        Self {
            params,
            server,
            counter: Rc::new(RefCell::new(0)),
        }
    }
}

impl Comparator for EncCmp {
    type Item = EncItem;
    type Aux = ();

    fn compare(&self, vs: &mut [Self::Item], i: usize, j: usize) {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let fft = fft.as_view();
        let mut stack = DynStack::new(&mut mem);

        let min_value =
            self.server
                .borrow()
                .min_with_fft(&vs[i].value, &vs[j].value, fft, &mut stack);
        let min_class = self.server.borrow().arg_min_with_fft(
            &vs[i].value,
            &vs[j].value,
            &vs[i].class,
            &vs[j].class,
            fft,
            &mut stack,
        );

        let mut max_value = self.server.borrow().raw_add(&vs[i].value, &vs[j].value);
        self.server
            .borrow()
            .raw_sub_assign(&mut max_value, &min_value);

        let mut max_class = self.server.borrow().raw_add(&vs[i].class, &vs[j].class);
        self.server
            .borrow()
            .raw_sub_assign(&mut max_class, &min_class);

        vs[i] = EncItem::new(min_value, min_class);
        vs[j] = EncItem::new(max_value, max_class);
        *self.counter.borrow_mut() += 1;
    }

    fn swap(&self, vs: &mut [Self::Item], i: usize, j: usize) {
        vs.swap(i, j);
    }

    fn compare_count(&self) -> usize {
        *self.counter.borrow()
    }
}

pub trait AsyncComparator: Sync + Send {
    type Item: Sync + Send;
    type Aux; // auxiliary information, e.g., FFT context

    fn compare(&self, a: &Self::Item, b: &Self::Item);
    fn swap(&self, a: &Self::Item, b: &Self::Item);
    fn compare_count(&self) -> usize;
}

pub struct AsyncClearComparator {
    counter: Arc<Mutex<usize>>,
    do_count: bool,
}

impl AsyncClearComparator {
    pub fn new() -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            do_count: false,
        }
    }
}

impl Default for AsyncClearComparator {
    fn default() -> Self {
        AsyncClearComparator::new()
    }
}

impl AsyncComparator for AsyncClearComparator {
    type Item = Arc<Mutex<u64>>;
    type Aux = ();

    fn compare(&self, a: &Self::Item, b: &Self::Item) {
        let a = a.clone();
        let b = b.clone();
        let mut a_value = a.lock().unwrap();
        let mut b_value = b.lock().unwrap();
        if *a_value > *b_value {
            std::mem::swap(&mut *a_value, &mut *b_value);
        }
        if self.do_count {
            let ctr = self.counter.clone();
            let mut curr = ctr.lock().unwrap();
            *curr += 1;
        }
    }

    fn swap(&self, a: &Self::Item, b: &Self::Item) {
        let a = a.clone();
        let b = b.clone();
        let mut a_guard = a.lock().unwrap();
        let mut b_guard = b.lock().unwrap();
        std::mem::swap(&mut *a_guard, &mut *b_guard);
    }

    fn compare_count(&self) -> usize {
        *self.counter.lock().unwrap()
    }
}

pub struct AsyncEncComparator {
    server: Arc<RwLock<KnnServer>>,
    params: Parameters,
    counter: Arc<Mutex<usize>>,
    do_count: bool,
}

impl AsyncEncComparator {
    pub fn new(server: Arc<RwLock<KnnServer>>, params: Parameters) -> Self {
        Self {
            server,
            params,
            counter: Arc::new(Mutex::new(0)),
            do_count: false,
        }
    }
}

impl AsyncComparator for AsyncEncComparator {
    type Item = Arc<Mutex<EncItem>>;
    type Aux = ();

    fn compare(&self, a: &Self::Item, b: &Self::Item) {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let fft = fft.as_view();
        let mut stack = DynStack::new(&mut mem);

        let a = a.clone();
        let b = b.clone();
        let mut a_guard = a.lock().unwrap();
        let mut b_guard = b.lock().unwrap();
        let server = self.server.clone();
        let server_guard = server.read().unwrap();

        let min_value = server_guard.min_with_fft(&a_guard.value, &b_guard.value, fft, &mut stack);
        let min_class = server_guard.arg_min_with_fft(
            &a_guard.value,
            &b_guard.value,
            &a_guard.class,
            &b_guard.class,
            fft,
            &mut stack,
        );

        let mut max_value = server_guard.raw_add(&a_guard.value, &b_guard.value);
        server_guard.raw_sub_assign(&mut max_value, &min_value);

        let mut max_class = server_guard.raw_add(&a_guard.class, &b_guard.class);
        server_guard.raw_sub_assign(&mut max_class, &min_class);

        *a_guard = EncItem::new(min_value, min_class);
        *b_guard = EncItem::new(max_value, max_class);

        if self.do_count {
            let ctr = self.counter.clone();
            let mut curr = ctr.lock().unwrap();
            *curr += 1;
        }
    }

    fn swap(&self, a: &Self::Item, b: &Self::Item) {
        let a = a.clone();
        let b = b.clone();
        let mut a_guard = a.lock().unwrap();
        let mut b_guard = b.lock().unwrap();
        std::mem::swap(&mut *a_guard, &mut *b_guard);
    }

    fn compare_count(&self) -> usize {
        *self.counter.lock().unwrap()
    }
}
