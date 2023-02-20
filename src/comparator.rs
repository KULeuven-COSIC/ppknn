use std::cmp::Ord;
use tfhe::shortint::prelude::*;
use tfhe::shortint::server_key::Accumulator;

pub trait Cmp {
    type Item;
    // NOTE: we can remove mut if we
    // put a mutex on every element
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

    fn inner(&self) -> &[T] {
        &self.vs
    }
}

pub struct EncCmp {
    cmp_count: usize,
    vs: Vec<Ciphertext>,
    params: Parameters,
    server_key: ServerKey,
    cmp_acc: Accumulator,
}

impl EncCmp {
    pub fn boxed(vs: Vec<Ciphertext>, params: &Parameters, server_key: ServerKey) -> Box<Self> {
        let modulus = params.message_modulus.0 as u64;
        let cmp_acc = server_key.generate_accumulator_bivariate(|x, y| x.min(y) % modulus);
        Box::new(Self {
            cmp_count: 0,
            vs,
            params: params.clone(),
            server_key,
            cmp_acc,
        })
    }

    pub fn print_params(&self) {
        println!("{:?}", self.params)
    }
}

impl Cmp for EncCmp {
    type Item = Ciphertext;

    fn cmp_at(&mut self, i: usize, j: usize) {
        // i gets the ct_res
        // j gets the other ciphertext
        let ct_res = self.server_key.keyswitch_programmable_bootstrap_bivariate(
            &self.vs[i],
            &self.vs[j],
            &self.cmp_acc,
        );
        let mut other = self.server_key.unchecked_add(&self.vs[i], &self.vs[j]);
        self.server_key.unchecked_sub_assign(&mut other, &ct_res);

        self.vs[i] = ct_res;
        self.vs[j] = other;
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.vs.swap(i, j);
    }

    fn split_at(&self, mid: usize) -> (&[Ciphertext], &[Ciphertext]) {
        self.vs.split_at(mid)
    }

    fn len(&self) -> usize {
        self.vs.len()
    }

    fn cmp_count(&self) -> usize {
        self.cmp_count
    }

    fn inner(&self) -> &[Ciphertext] {
        &self.vs
    }
}
