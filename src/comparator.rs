use crate::server::KnnServer;
use crate::setup_polymul_fft;
use dyn_stack::DynStack;
use std::cell::RefCell;
use std::cmp::{Ord, Ordering};
use std::fmt;
use std::rc::Rc;
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
    // NOTE: we can remove mut if we
    // use interior mutability
    fn cmp_at(&mut self, i: usize, j: usize);
    fn swap(&mut self, i: usize, j: usize);
    fn split_at(&self, mid: usize) -> (&[Self::Item], &[Self::Item]);
    fn len(&self) -> usize;
    fn cmp_count(&self) -> usize;
    fn inner(&self) -> &[Self::Item];
}

pub struct ClearCmp<T: Ord + Clone> {
    cmp_count: usize,
    vs: Vec<T>,
}

impl<T: Ord + Clone> ClearCmp<T> {
    /// Create a plaintext vector that implements `Comparator`.
    pub fn new(vs: Vec<T>) -> Self {
        Self { cmp_count: 0, vs }
    }

    /// Create a plaintext vector that implements `Comparator`
    /// and output a boxed `Comparator`.
    pub fn boxed(vs: Vec<T>) -> Box<Self> {
        Box::new(Self { cmp_count: 0, vs })
    }
}

impl<T: Ord + Clone> Comparator for ClearCmp<T> {
    type Item = T;

    fn cmp_at(&mut self, i: usize, j: usize) {
        self.cmp_count += 1;
        if self.vs[i] > self.vs[j] {
            self.vs.swap(i, j);
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.vs.swap(i, j);
    }

    fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        self.vs.split_at(mid)
    }

    fn len(&self) -> usize {
        self.vs.len()
    }

    fn cmp_count(&self) -> usize {
        self.cmp_count
    }

    fn inner(&self) -> &[T] {
        &self.vs
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
    cmp_count: usize,
    vs: Vec<EncItem>,
    params: Parameters,
    server: Rc<RefCell<KnnServer>>,
}

impl EncCmp {
    /// Create an encrypted vector that implements `Comparator`.
    /// A reference to `KnnServer` is needed because it has the cryptography context.
    /// The output is boxed.
    pub fn boxed(
        vs: Vec<EncItem>,
        params: Parameters,
        server: Rc<RefCell<KnnServer>>,
    ) -> Box<Self> {
        Box::new(Self {
            cmp_count: 0,
            vs,
            params,
            server,
        })
    }
}

impl Comparator for EncCmp {
    type Item = EncItem;

    fn cmp_at(&mut self, i: usize, j: usize) {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let fft = fft.as_view();
        let mut stack = DynStack::new(&mut mem);

        let min_value = self.server.borrow().min_with_fft(
            &self.vs[i].value,
            &self.vs[j].value,
            fft,
            &mut stack,
        );
        let min_class = self.server.borrow().arg_min_with_fft(
            &self.vs[i].value,
            &self.vs[j].value,
            &self.vs[i].class,
            &self.vs[j].class,
            fft,
            &mut stack,
        );

        let mut max_value = self
            .server
            .borrow()
            .raw_add(&self.vs[i].value, &self.vs[j].value);
        self.server
            .borrow()
            .raw_sub_assign(&mut max_value, &min_value);

        let mut max_class = self
            .server
            .borrow()
            .raw_add(&self.vs[i].class, &self.vs[j].class);
        self.server
            .borrow()
            .raw_sub_assign(&mut max_class, &min_class);

        self.vs[i] = EncItem::new(min_value, min_class);
        self.vs[j] = EncItem::new(max_value, max_class);
        self.cmp_count += 1;
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.vs.swap(i, j);
    }

    fn split_at(&self, mid: usize) -> (&[EncItem], &[EncItem]) {
        self.vs.split_at(mid)
    }

    fn len(&self) -> usize {
        self.vs.len()
    }

    fn cmp_count(&self) -> usize {
        self.cmp_count
    }

    fn inner(&self) -> &[EncItem] {
        &self.vs
    }
}
