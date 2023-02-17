use std::cmp::Ord;

pub trait Comparator {
    type Item;
    // NOTE: we can remove mut if we
    // put a mutex on every element
    fn cmp_at(&mut self, i: usize, j: usize);
    fn swap(&mut self, i: usize, j: usize);
    fn split_at(&self, mid: usize) -> (&[Self::Item], &[Self::Item]);
    fn len(&self) -> usize;
    fn comparisons(&self) -> usize;
}

struct ClearComparator<T: Ord> {
    comp_count: usize,
    vs: Vec<T>,
}

impl<T: Ord> ClearComparator<T> {
    pub fn new(vs: Vec<T>) -> Self {
        Self { comp_count: 0, vs }
    }
}

impl <T: Ord>Comparator for ClearComparator<T> {
    type Item = T;

    fn cmp_at(&mut self, i: usize, j: usize) {
        self.comp_count += 1;
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

    fn comparisons(&self) -> usize {
        self.comp_count
    }
}

