use crate::comparator::Comparator;
use crate::AsyncComparator;
use rayon;
use std::cmp;
use std::collections::HashMap;

fn build_local_index_map(ix: &[usize], jx: &[usize]) -> HashMap<usize, usize> {
    let mut out = HashMap::with_capacity(ix.len() + jx.len());
    let mut i = 0usize;
    for x in ix {
        out.insert(*x, i);
        i += 1;
    }
    for y in jx {
        out.insert(*y, i);
        i += 1;
    }
    out
}

fn split_indices<T>(vs: &[T], k: usize, verbose: bool) -> Vec<Vec<usize>> {
    let len = vs.len();
    let mut out = vec![];
    let chunk_size = if k == 1 {
        2
    } else {
        2usize.pow((k as f64).log2().ceil() as u32)
    };
    let chunks = len / chunk_size;
    for i in 0..chunks {
        let v: Vec<_> = (i * chunk_size..i * chunk_size + chunk_size).collect();
        out.push(v);
    }

    let rem = len % chunk_size;
    if verbose {
        println!(
            "[split_indices] k={}, len={}, chunks={}, chunk_size={}, rem={}",
            k, len, chunks, chunk_size, rem
        );
    }
    if rem != 0 {
        let v: Vec<_> = (len - rem..len).collect();
        assert_eq!(v.len(), rem);
        out.push(v);
    }
    out
}

fn odd_indices(indices: &[usize]) -> Vec<usize> {
    let new_indices = indices.split_first().unwrap().1;
    even_indices(new_indices)
}

fn even_indices(indices: &[usize]) -> Vec<usize> {
    let mut out = vec![];
    for i in (0..indices.len()).step_by(2) {
        out.push(indices[i]);
    }
    // if tournament method is used this step is redundant
    /*
    let expected_len = if out.len() > self.k {
        self.k
    } else {
        out.len()
    };
    out.split_at(expected_len).0.try_into().unwrap()
    */
    out
}

pub struct BatcherSort<CMP> {
    k: usize,
    cmp: CMP,
    verbose: bool,
}

impl<CMP: Comparator> BatcherSort<CMP> {
    /// Create an instance of the truncated Batcher's odd-even network
    /// where the the output length is `k`.
    pub fn new_k(k: usize, cmp: CMP, verbose: bool) -> Self {
        Self { k, cmp, verbose }
    }

    /// Run the sorting network.
    pub fn sort(&self, vs: &mut [CMP::Item]) {
        if vs.len() <= 4 {
            // for lengths lower or equal to 4,
            // we cannot split them more than 2,
            // so just call `sort_rec` directly.
            let chunks: Vec<_> = (0..vs.len()).collect();
            self.sort_rec(vs, &chunks);
        } else {
            let chunks = split_indices(vs, self.k, self.verbose);
            for chunk in &chunks {
                self.sort_rec(vs, chunk);
            }
            self.tournament_merge(vs, chunks);
        }
    }

    fn sort_rec(&self, vs: &mut [CMP::Item], indices: &[usize]) {
        if self.verbose {
            println!("[sort_rec begin] indices={:?}", indices);
        }
        // sort every chunk recursively
        if indices.len() > 1 {
            let n = indices.len() / 2;
            let m = indices.len() - n;
            self.sort_rec(vs, &indices[0..n]);
            self.sort_rec(vs, &indices[n..n + m]);

            // let indices: Vec<_> = (start..start + len).collect();
            let (ix_full, jx_full) = indices.split_at(n);

            let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
            let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
            self.merge_rec(vs, ix, jx, self.k);
        }
        if self.verbose {
            println!("[sort_rec exit] indices={:?}", indices);
        }
    }

    /// Execute the merge step between two halves
    /// where the first half has length `n/2` (indices `0..n/2`)
    /// and the second half has length `n - n/2` (indices `n/2..n`).
    /// We assume the two arrays we wish to merge are already sorted.
    pub fn merge(&self, vs: &mut [CMP::Item]) {
        let n = vs.len() / 2;
        let m = vs.len() - n;

        let ix_full: Vec<_> = (0..n).collect();
        let jx_full: Vec<_> = (n..n + m).collect();
        let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
        let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
        self.merge_rec(vs, ix, jx, self.k)
    }

    fn tournament_merge(&self, vs: &mut [CMP::Item], index_sets: Vec<Vec<usize>>) {
        if index_sets.len() == 1 || index_sets.is_empty() {
            return;
        }

        // merge every pair of index_sets and
        // ignore the last one if the set size is odd
        let mut new_index_sets: Vec<Vec<usize>> = Vec::with_capacity(index_sets.len() / 2);
        for i in 0..index_sets.len() / 2 {
            let len_left = if index_sets[i * 2].len() > self.k {
                self.k
            } else {
                index_sets[i * 2].len()
            };
            let len_right = if index_sets[i * 2 + 1].len() > self.k {
                self.k
            } else {
                index_sets[i * 2 + 1].len()
            };
            // the output length is the minimum of `k` and
            // the total number of values in each chunk
            let output_len = (self.k as f64).min(len_left as f64 + len_right as f64) as usize;
            self.merge_rec(
                vs,
                &index_sets[i * 2][0..len_left],
                &index_sets[i * 2 + 1][0..len_right],
                output_len,
            );

            // build a new set that combines the two old ones
            let mut s = index_sets[i * 2].clone();
            s.extend(index_sets[i * 2 + 1].iter());
            new_index_sets.push(s);
        }
        if index_sets.len() % 2 == 1 {
            new_index_sets.push(index_sets.last().unwrap().clone());
        }
        self.tournament_merge(vs, new_index_sets);
    }

    fn merge_rec(&self, vs: &mut [CMP::Item], ix: &[usize], jx: &[usize], output_len: usize) {
        if self.verbose {
            println!("[merge begin] ix={:?}, jx={:?}", ix, jx);
        }
        let nm = ix.len() * jx.len();
        if nm > 1 {
            let even_ix = even_indices(ix);
            let even_jx = even_indices(jx);
            let odd_ix = odd_indices(ix);
            let odd_jx = odd_indices(jx);

            let odd_output_len = ((output_len as f64 - 1.) / 2.).ceil() as usize;
            let even_output_len = output_len - odd_output_len;
            self.merge_rec(vs, &even_ix, &even_jx, even_output_len);
            self.merge_rec(vs, &odd_ix, &odd_jx, odd_output_len);

            let even_all = [even_ix, even_jx].concat();
            let odd_all = [odd_ix, odd_jx].concat();
            let tmp = ((even_all.len() as f64 / 2f64).floor()
                + (odd_all.len() as f64 / 2f64).floor()) as usize;
            let w_max = if even_all.len() % 2 == 0 && odd_all.len() % 2 == 0 {
                tmp - 1
            } else {
                tmp
            };

            // maps the global index to the local index
            let local_index_map = build_local_index_map(ix, jx);
            for i in 0..w_max {
                // we need to compare the local index, not the global one
                // i.e., ix[0] is at local index 0, jx[0] is at local index |ix|
                if local_index_map[&odd_all[i]] < output_len
                    || local_index_map[&even_all[i + 1]] < output_len
                {
                    self.cmp.compare(vs, odd_all[i], even_all[i + 1]);
                }
            }

            // the final output is v1, w1, v2, w2...
            // correction needed if `|ix|` is odd
            if ix.len() % 2 == 1 {
                let end = if jx.len() % 2 == 0 {
                    jx.len()
                } else {
                    jx.len() - 1
                };
                for i in (0..end).step_by(2) {
                    self.cmp.swap(vs, jx[i], jx[i + 1]);
                }
            }
        } else if nm == 1 {
            self.cmp.compare(vs, ix[0], jx[0]);
        } else {
            // do nothing because we have 1 or 0 elements
        }
        if self.verbose {
            println!("[merge exit] ix={:?}, jx={:?}", ix, jx);
        }
    }

    /// Output the number of comparisons
    pub fn comparisons(&self) -> usize {
        self.cmp.compare_count()
    }
}

impl<CMP: AsyncComparator + Sync + Send> BatcherSort<CMP> {
    pub fn par_new_k(k: usize, cmp: CMP, verbose: bool) -> Self {
        // TODO can we use the new_k function?
        Self { k, cmp, verbose }
    }

    pub fn par_sort(&self, vs: &[CMP::Item]) {
        if vs.len() <= 4 {
            // for lengths lower or equal to 4,
            // we cannot split them more than 2,
            // so just call `sort_rec` directly.
            let chunks: Vec<_> = (0..vs.len()).collect();
            self.par_sort_rec(vs, &chunks);
        } else {
            let chunks = split_indices(vs, self.k, self.verbose);
            for chunk in &chunks {
                self.par_sort_rec(vs, chunk);
            }
            self.par_tournament_merge(vs, chunks);
        }
    }

    fn par_sort_rec(&self, vs: &[CMP::Item], indices: &[usize]) {
        if self.verbose {
            println!("[sort_rec begin] indices={:?}", indices);
        }
        // sort every chunk recursively
        if indices.len() > 1 {
            let n = indices.len() / 2;
            let m = indices.len() - n;
            rayon::join(
                || self.par_sort_rec(vs, &indices[0..n]),
                || self.par_sort_rec(vs, &indices[n..n + m]),
            );

            // let indices: Vec<_> = (start..start + len).collect();
            let (ix_full, jx_full) = indices.split_at(n);

            let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
            let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
            self.par_merge_rec(vs, ix, jx, self.k);
        }
        if self.verbose {
            println!("[sort_rec exit] indices={:?}", indices);
        }
    }

    /// Execute the merge step between two halves
    /// where the first half has length `n/2` (indices `0..n/2`)
    /// and the second half has length `n - n/2` (indices `n/2..n`).
    /// We assume the two arrays we wish to merge are already sorted.
    pub fn par_merge(&self, vs: &[CMP::Item]) {
        let n = vs.len() / 2;
        let m = vs.len() - n;

        let ix_full: Vec<_> = (0..n).collect();
        let jx_full: Vec<_> = (n..n + m).collect();
        let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
        let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
        self.par_merge_rec(vs, ix, jx, self.k)
    }

    fn par_tournament_merge(&self, vs: &[CMP::Item], index_sets: Vec<Vec<usize>>) {
        if index_sets.len() == 1 || index_sets.is_empty() {
            return;
        }

        // merge every pair of index_sets and
        // ignore the last one if the set size is odd
        let mut new_index_sets: Vec<Vec<usize>> = Vec::with_capacity(index_sets.len() / 2);
        for i in 0..index_sets.len() / 2 {
            let len_left = if index_sets[i * 2].len() > self.k {
                self.k
            } else {
                index_sets[i * 2].len()
            };
            let len_right = if index_sets[i * 2 + 1].len() > self.k {
                self.k
            } else {
                index_sets[i * 2 + 1].len()
            };
            // the output length is the minimum of `k` and
            // the total number of values in each chunk
            let output_len = (self.k as f64).min(len_left as f64 + len_right as f64) as usize;
            self.par_merge_rec(
                vs,
                &index_sets[i * 2][0..len_left],
                &index_sets[i * 2 + 1][0..len_right],
                output_len,
            );

            // build a new set that combines the two old ones
            let mut s = index_sets[i * 2].clone();
            s.extend(index_sets[i * 2 + 1].iter());
            new_index_sets.push(s);
        }
        if index_sets.len() % 2 == 1 {
            new_index_sets.push(index_sets.last().unwrap().clone());
        }
        self.par_tournament_merge(vs, new_index_sets);
    }

    fn par_merge_rec(&self, vs: &[CMP::Item], ix: &[usize], jx: &[usize], output_len: usize) {
        if self.verbose {
            println!("[merge begin] ix={:?}, jx={:?}", ix, jx);
        }
        let nm = ix.len() * jx.len();
        if nm > 1 {
            let even_ix = even_indices(ix);
            let even_jx = even_indices(jx);
            let odd_ix = odd_indices(ix);
            let odd_jx = odd_indices(jx);

            let odd_output_len = ((output_len as f64 - 1.) / 2.).ceil() as usize;
            let even_output_len = output_len - odd_output_len;
            rayon::join(
                || self.par_merge_rec(vs, &even_ix, &even_jx, even_output_len),
                || self.par_merge_rec(vs, &odd_ix, &odd_jx, odd_output_len),
            );

            let even_all = [even_ix, even_jx].concat();
            let odd_all = [odd_ix, odd_jx].concat();
            let tmp = ((even_all.len() as f64 / 2f64).floor()
                + (odd_all.len() as f64 / 2f64).floor()) as usize;
            let w_max = if even_all.len() % 2 == 0 && odd_all.len() % 2 == 0 {
                tmp - 1
            } else {
                tmp
            };

            // maps the global index to the local index
            let local_index_map = build_local_index_map(ix, jx);
            for i in 0..w_max {
                // we need to compare the local index, not the global one
                // i.e., ix[0] is at local index 0, jx[0] is at local index |ix|
                if local_index_map[&odd_all[i]] < output_len
                    || local_index_map[&even_all[i + 1]] < output_len
                {
                    self.cmp.compare(&vs[odd_all[i]], &vs[even_all[i + 1]]);
                    // self.vs.cmp_at(odd_all[i], even_all[i + 1]);
                }
            }

            // the final output is v1, w1, v2, w2...
            // correction needed if `|ix|` is odd
            if ix.len() % 2 == 1 {
                let end = if jx.len() % 2 == 0 {
                    jx.len()
                } else {
                    jx.len() - 1
                };
                for i in (0..end).step_by(2) {
                    self.cmp.swap(&vs[jx[i]], &vs[jx[i + 1]]);
                }
            }
        } else if nm == 1 {
            self.cmp.compare(&vs[ix[0]], &vs[jx[0]]);
        } else {
            // do nothing because we have 1 or 0 elements
        }
        if self.verbose {
            println!("[merge exit] ix={:?}, jx={:?}", ix, jx);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::comparator::AsyncClearComparator;
    use crate::comparator::ClearCmp;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use std::sync::{Arc, Mutex};

    fn helper_merge(vs: &mut [i32]) -> usize {
        helper_merge_k(vs, vs.len())
    }

    fn helper_merge_k(vs: &mut [i32], k: usize) -> usize {
        let cmp = ClearCmp::<i32>::new();
        let batcher = BatcherSort::new_k(k, cmp, true);
        batcher.merge(vs);
        batcher.comparisons()
    }

    #[test]
    fn test_merge_2() {
        {
            let mut vs = vec![2, 1];
            let comparisons = helper_merge(&mut vs);
            assert_eq!(vec![1, 2], vs);
            assert_eq!(1, comparisons);
        }
        {
            let k = 1;
            let mut vs = vec![2, 1];
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1], vs.split_at(k).0);
            assert_eq!(1, comparisons);
        }
    }

    #[test]
    fn test_merge_3() {
        {
            let mut vs = vec![1, 5, 2];
            let comparisons = helper_merge(&mut vs);
            assert_eq!(vec![1, 2, 5], vs);
            assert_eq!(2, comparisons);
        }
        {
            let mut vs = vec![2, 1, 5];
            let k = 1;
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1], vs.split_at(k).0);
            assert_eq!(1, comparisons);
        }
    }

    #[test]
    fn test_merge_4() {
        {
            let mut vs = vec![1, 5, 2, 4];
            let comparisons = helper_merge(&mut vs);
            assert_eq!(vec![1, 2, 4, 5], vs);
            assert_eq!(3, comparisons);
        }
        {
            let mut vs = vec![2, 5, 1, 4];
            let k = 1;
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1], vs.split_at(k).0);
            assert_eq!(1, comparisons);
        }
    }

    #[test]
    fn test_merge_5() {
        let mut vs = vec![1, 5, 6, 2, 4];
        let _ = helper_merge(&mut vs);
        assert_eq!(vec![1, 2, 4, 5, 6], vs);
    }

    #[test]
    fn test_merge_8() {
        {
            let mut vs = vec![1, 5, 6, 7, 2, 3, 4, 5];
            let comparisons = helper_merge(&mut vs);
            assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], vs);
            assert_eq!(9, comparisons);
        }
        {
            let mut vs = vec![1, 5, 6, 7, 2, 3, 4, 5];
            let k = 1;
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1], vs.split_at(k).0);
            assert_eq!(1, comparisons);
        }
        {
            let mut vs = vec![1, 5, 6, 7, 2, 3, 4, 5];
            let k = 2;
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1, 2], vs.split_at(k).0);
            assert_eq!(3, comparisons);
        }
        {
            let mut vs = vec![1, 5, 6, 7, 2, 3, 4, 5];
            let k = 4;
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1, 2, 3, 4], vs.split_at(k).0);
            assert_eq!(8, comparisons);
        }
        {
            let mut vs = vec![1, 5, 6, 7, 2, 3, 4, 5];
            let k = 5;
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1, 2, 3, 4, 5], vs.split_at(k).0);
            assert_eq!(8, comparisons);
        }
        {
            let mut vs = vec![1, 5, 6, 7, 2, 3, 4, 5];
            let k = 6;
            let comparisons = helper_merge_k(&mut vs, k);
            assert_eq!(vec![1, 2, 3, 4, 5, 5], vs.split_at(k).0);
            assert_eq!(9, comparisons);
        }
    }

    #[test]
    fn test_merge_10() {
        let mut vs = vec![2, 4, 6, 8, 10, 1, 3, 5, 7, 9];
        let k = 5;
        let comparisons = helper_merge_k(&mut vs, k);
        assert_eq!(vec![1, 2, 3, 4, 5], vs.split_at(k).0);
        assert_eq!(10, comparisons);
    }

    fn helper_sort(mut xs: Vec<i32>) -> usize {
        let k = xs.len();
        helper_sort_k(xs, k)
    }

    fn helper_sort_k(mut actual: Vec<i32>, k: usize) -> usize {
        let mut expected = actual.clone();
        expected.sort();

        let cmp = ClearCmp::<i32>::new();
        let batcher = BatcherSort::new_k(k, cmp, true);
        batcher.sort(&mut actual);
        assert_eq!(actual.split_at(k).0, expected.split_at(k).0);

        batcher.comparisons()
    }

    #[test]
    fn test_sort() {
        helper_sort(vec![5, 1, 6, 2]);
        helper_sort(vec![5, 4, 3, 2, 1]);
        helper_sort(vec![7, 6, 5, 4, 3, 2, 1]);
        helper_sort(vec![1, 5, 6, 7, 2, 4, 3, 5]);
        helper_sort_k(vec![5, 1, 6, 7], 1);
        assert_eq!(5, helper_sort_k(vec![0; 4], 2));
        assert_eq!(13, helper_sort_k(vec![0; 8], 2));
        assert_eq!(29, helper_sort_k(vec![0; 16], 2));
        assert_eq!(20, helper_sort_k(vec![0; 10], 3));
    }

    #[quickcheck]
    fn prop_sort(mut xs: Vec<usize>) -> TestResult {
        if xs.len() > 20 {
            return TestResult::discard();
        }
        let mut sorted = xs.clone();
        sorted.sort();

        let cmp = ClearCmp::<usize>::new();
        let batcher = BatcherSort::new_k(xs.len(), cmp, false);
        batcher.sort(&mut xs);

        TestResult::from_bool(xs == sorted)
    }

    #[quickcheck]
    fn prop_sort_async(xs: Vec<u64>) -> TestResult {
        if xs.len() > 20 {
            return TestResult::discard();
        }
        let mut sorted = xs.clone();
        sorted.sort();

        let a = AsyncClearComparator::new();
        let batcher = BatcherSort::par_new_k(xs.len(), a, false);

        let async_xs: Vec<_> = xs.into_iter().map(|x| Arc::new(Mutex::new(x))).collect();
        batcher.par_sort(&async_xs);

        let actual: Vec<_> = async_xs.into_iter().map(|x| *x.lock().unwrap()).collect();
        TestResult::from_bool(actual == sorted)
    }

    #[quickcheck]
    fn prop_sort_k(mut xs: Vec<u16>, k: usize) -> TestResult {
        if xs.len() > 20 {
            return TestResult::discard();
        }

        if k > xs.len() {
            return TestResult::discard();
        }

        let mut sorted = xs.clone();
        sorted.sort();

        let cmp = ClearCmp::<u16>::new();
        let batcher = BatcherSort::new_k(k, cmp, false);
        batcher.sort(&mut xs);

        TestResult::from_bool(xs.split_at(k).0 == sorted.split_at(k).0)
    }

    #[quickcheck]
    fn prop_sort_k_5(mut xs: Vec<u16>) -> TestResult {
        if xs.len() > 5000 || xs.is_empty() {
            return TestResult::discard();
        }
        let k = 5usize.min(xs.len());

        let mut sorted = xs.clone();
        sorted.sort();

        let cmp = ClearCmp::<u16>::new();
        let batcher = BatcherSort::new_k(k, cmp, false);
        batcher.sort(&mut xs);

        TestResult::from_bool(xs.split_at(k).0 == sorted.split_at(k).0)
    }

    #[quickcheck]
    fn prop_sort_k_2(mut xs: Vec<u16>) -> TestResult {
        if xs.len() > 5000 || xs.is_empty() {
            return TestResult::discard();
        }
        let k = 2usize.min(xs.len());

        let mut sorted = xs.clone();
        sorted.sort();

        let cmp = ClearCmp::<u16>::new();
        let batcher = BatcherSort::new_k(k, cmp, false);
        batcher.sort(&mut xs);

        TestResult::from_bool(xs.split_at(k).0 == sorted.split_at(k).0)
    }
}
