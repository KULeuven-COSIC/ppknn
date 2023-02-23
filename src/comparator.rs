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

pub struct EncItem {
    pub value: Ciphertext,
    pub class: Ciphertext,
}

impl EncItem {
    pub fn new(value: Ciphertext, class: Ciphertext) -> Self {
        Self { value, class }
    }

    pub fn decrypt(&self, client_key: &ClientKey) -> (u64, u64) {
        let modulus = client_key.parameters.message_modulus.0 as u64;
        (
            client_key.decrypt(&self.value) % modulus,
            client_key.decrypt(&self.class) % modulus,
        )
    }
}

pub struct EncCmp {
    cmp_count: usize,
    vs: Vec<EncItem>,
    params: Parameters,
    server_key: ServerKey,
    acc: Accumulator,
}

impl EncCmp {
    pub fn boxed(vs: Vec<EncItem>, params: &Parameters, server_key: ServerKey) -> Box<Self> {
        let modulus = params.message_modulus.0 as u64;
        let acc = server_key.generate_accumulator(|x| if x >= modulus / 2 { 1 } else { 0 });
        Box::new(Self {
            cmp_count: 0,
            vs,
            params: params.clone(),
            server_key,
            acc,
        })
    }

    pub fn print_params(&self) {
        println!("{:?}", self.params)
    }
}

impl Cmp for EncCmp {
    type Item = EncItem;

    fn cmp_at(&mut self, i: usize, j: usize) {
        let diff = self
            .server_key
            .unchecked_sub(&self.vs[i].value, &self.vs[j].value);
        let cmp_res = self
            .server_key
            .keyswitch_programmable_bootstrap(&diff, &self.acc);
        // let cmp_res = self.server_key.unchecked_less(&self.vs[i].value, &self.vs[j].value);

        // vs[j] + cmp_res * (vs[i] - vs[j])
        let mut smaller = self
            .server_key
            .unchecked_sub(&self.vs[i].value, &self.vs[j].value);
        self.server_key
            .unchecked_mul_lsb_assign(&mut smaller, &cmp_res);
        self.server_key
            .unchecked_add_assign(&mut smaller, &self.vs[j].value);

        let mut bigger = self
            .server_key
            .unchecked_add(&self.vs[i].value, &self.vs[j].value);
        self.server_key.unchecked_sub_assign(&mut bigger, &smaller);

        // j + cmp_res * (i - j)
        let mut smaller_class = self
            .server_key
            .unchecked_sub(&self.vs[i].class, &self.vs[j].class);
        self.server_key
            .unchecked_mul_lsb_assign(&mut smaller_class, &cmp_res);
        self.server_key
            .unchecked_add_assign(&mut smaller_class, &self.vs[j].class);

        let mut bigger_class = self
            .server_key
            .unchecked_add(&self.vs[i].class, &self.vs[j].class);
        self.server_key
            .unchecked_sub_assign(&mut bigger_class, &smaller_class);

        self.vs[i] = EncItem::new(smaller, smaller_class);
        self.vs[j] = EncItem::new(bigger, bigger_class);
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
