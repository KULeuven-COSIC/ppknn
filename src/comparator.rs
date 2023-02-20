use std::cmp::Ord;

pub trait Cmp {
    type Item;
    // NOTE: we can remove mut if we
    // put a mutex on every element
    fn cmp_at(&mut self, i: usize, j: usize);
    fn swap(&mut self, i: usize, j: usize);
    fn split_at(&self, mid: usize) -> (&[Self::Item], &[Self::Item]);
    fn len(&self) -> usize;
    fn cmp_count(&self) -> usize;
    fn inner(&self) -> Vec<Self::Item>;
}

pub struct ClearCmp<T: Ord + Clone> {
    cmp_count: usize,
    vs: Vec<T>,
}

impl<T: Ord + Clone> ClearCmp<T> {
    pub fn new(vs: Vec<T>) -> Self {
        Self { cmp_count: 0, vs }
    }

    pub fn boxed(vs: Vec<T>) -> Box<Self> {
        Box::new(Self { cmp_count: 0, vs })
    }
}

impl<T: Ord + Clone> Cmp for ClearCmp<T> {
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

    fn inner(&self) -> Vec<T> {
        self.vs.clone()
    }
}
