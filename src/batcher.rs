use crate::comparator::Comparator;
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

pub struct BatcherSort<T> {
    pub vs: Box<dyn Comparator<Item = T>>,
    k: usize,
    verbose: bool,
}

impl<T> BatcherSort<T> {
    /// Create an instance the full Batcher's odd-even network
    /// where the output length is set to the length of `vs`.
    pub fn new(vs: Box<dyn Comparator<Item = T>>) -> Self {
        let k = vs.len();
        Self {
            vs,
            k,
            verbose: false,
        }
    }

    /// Create an instance of the truncated Batcher's odd-even network
    /// where the the output length is `k`.
    pub fn new_k(vs: Box<dyn Comparator<Item = T>>, k: usize) -> Self {
        let k = k.min(vs.len());
        Self {
            vs,
            k,
            verbose: false,
        }
    }

    /// Run the sorting network.
    pub fn sort(&mut self) {
        if self.vs.len() <= 4 {
            // for lengths lower or equal to 4,
            // we cannot split them more than 2,
            // so just call `sort_rec` directly.
            let chunks: Vec<_> = (0..self.vs.len()).collect();
            self.sort_rec(&chunks);
        } else {
            let chunks = self.split_indices();
            for chunk in &chunks {
                self.sort_rec(chunk);
            }
            self.tournament_merge(chunks);
        }
    }

    fn split_indices(&self) -> Vec<Vec<usize>> {
        let len = self.vs.len();
        let mut out = vec![];
        let chunk_size = if self.k == 1 {
            2
        } else {
            2usize.pow((self.k as f64).log2().ceil() as u32)
        };
        let chunks = len / chunk_size;
        for i in 0..chunks {
            let v: Vec<_> = (i * chunk_size..i * chunk_size + chunk_size).collect();
            out.push(v);
        }

        let rem = len % chunk_size;
        if self.verbose {
            println!(
                "[split_indices] k={}, len={}, chunks={}, chunk_size={}, rem={}",
                self.k, len, chunks, chunk_size, rem
            );
        }
        if rem != 0 {
            let v: Vec<_> = (len - rem..len).collect();
            assert_eq!(v.len(), rem);
            out.push(v);
        }
        out
    }

    fn sort_rec(&mut self, indices: &[usize]) {
        if self.verbose {
            println!("[sort_rec begin] indices={:?}", indices);
        }
        // sort every chunk recursively
        if indices.len() > 1 {
            let n = indices.len() / 2;
            let m = indices.len() - n;
            self.sort_rec(&indices[0..n]);
            self.sort_rec(&indices[n..n + m]);

            // let indices: Vec<_> = (start..start + len).collect();
            let (ix_full, jx_full) = indices.split_at(n);

            let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
            let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
            self.merge_rec(&ix, &jx, self.k);
        }
        if self.verbose {
            println!("[sort_rec exit] indices={:?}", indices);
        }
    }

    /// Execute the merge step between two halves
    /// where the first half has length `n/2` (indices `0..n/2`)
    /// and the second half has length `n - n/2` (indices `n/2..n`).
    /// We assume the two arrays we wish to merge are already sorted.
    pub fn merge(&mut self) {
        let n = self.vs.len() / 2;
        let m = self.vs.len() - n;

        let ix_full: Vec<_> = (0..n).collect();
        let jx_full: Vec<_> = (n..n + m).collect();
        let (ix, _) = ix_full.split_at(cmp::min(ix_full.len(), self.k));
        let (jx, _) = jx_full.split_at(cmp::min(jx_full.len(), self.k));
        self.merge_rec(&ix, &jx, self.k)
    }

    fn tournament_merge(&mut self, index_sets: Vec<Vec<usize>>) {
        if index_sets.len() == 1 || index_sets.len() == 0 {
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
        self.tournament_merge(new_index_sets);
    }

    fn merge_rec(&mut self, ix: &[usize], jx: &[usize], output_len: usize) {
        if self.verbose {
            println!("[merge begin] ix={:?}, jx={:?}", ix, jx);
        }
        let nm = ix.len() * jx.len();
        if nm > 1 {
            let even_ix = self.even_indices(ix);
            let even_jx = self.even_indices(jx);
            let odd_ix = self.odd_indices(ix);
            let odd_jx = self.odd_indices(jx);

            let odd_output_len = ((output_len as f64 - 1.) / 2.).ceil() as usize;
            let even_output_len = output_len - odd_output_len;
            self.merge_rec(&even_ix, &even_jx, even_output_len);
            self.merge_rec(&odd_ix, &odd_jx, odd_output_len);

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
                    self.vs.cmp_at(odd_all[i], even_all[i + 1]);
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
                    self.vs.swap(jx[i], jx[i + 1]);
                }
            }
        } else if nm == 1 {
            self.vs.cmp_at(ix[0], jx[0]);
        } else {
            // do nothing because we have 1 or 0 elements
        }
        if self.verbose {
            println!("[merge exit] ix={:?}, jx={:?}", ix, jx);
        }
    }

    /// Output the number of comparisons
    pub fn comparisons(&self) -> usize {
        self.vs.cmp_count()
    }

    pub fn inner(&self) -> &[T] {
        self.vs.inner()
    }

    fn odd_indices(&self, indices: &[usize]) -> Vec<usize> {
        let new_indices = indices.split_first().unwrap().1;
        self.even_indices(new_indices)
    }

    fn even_indices(&self, indices: &[usize]) -> Vec<usize> {
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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::comparator::ClearCmp;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[test]
    fn test_merge_2() {
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![2, 1]));
            batcher.merge();
            assert_eq!(vec![1, 2], batcher.inner());
            assert_eq!(1, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![2, 1]), k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_3() {
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![1, 5, 2]));
            batcher.merge();
            assert_eq!(vec![1, 2, 5], batcher.inner());
            assert_eq!(2, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![2, 1, 5]), k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_4() {
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![1, 5, 2, 4]));
            batcher.merge();
            assert_eq!(vec![1, 2, 4, 5], batcher.inner());
            assert_eq!(3, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![2, 5, 1, 4]), k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_5() {
        let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![1, 5, 6, 2, 4]));
        batcher.merge();
        assert_eq!(vec![1, 2, 4, 5, 6], batcher.inner());
    }

    #[test]
    fn test_merge_8() {
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![1, 5, 6, 7, 2, 3, 4, 5]));
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.inner());
            assert_eq!(9, batcher.comparisons());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![1, 5, 6, 7, 2, 3, 4, 5]), k);
            batcher.merge();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
            assert_eq!(1, batcher.comparisons());
        }
        {
            let k = 2;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![1, 5, 6, 7, 2, 3, 4, 5]), k);
            batcher.merge();
            assert_eq!(vec![1, 2], batcher.vs.split_at(k).0);
            assert_eq!(3, batcher.comparisons());
        }
        {
            let k = 4;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![1, 5, 6, 7, 2, 3, 4, 5]), k);
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4], batcher.vs.split_at(k).0);
            assert_eq!(8, batcher.comparisons());
        }
        {
            let k = 5;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![1, 5, 6, 7, 2, 3, 4, 5]), k);
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4, 5], batcher.vs.split_at(k).0);
            assert_eq!(8, batcher.comparisons());
        }
        {
            let k = 6;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![1, 5, 6, 7, 2, 3, 4, 5]), k);
            batcher.merge();
            assert_eq!(vec![1, 2, 3, 4, 5, 5], batcher.vs.split_at(k).0);
            assert_eq!(9, batcher.comparisons());
        }
    }

    #[test]
    fn test_merge_10() {
        let k = 5;
        let mut batcher =
            BatcherSort::new_k(ClearCmp::boxed(vec![2, 4, 6, 8, 10, 1, 3, 5, 7, 9]), k);
        batcher.merge();
        assert_eq!(vec![1, 2, 3, 4, 5], batcher.vs.split_at(k).0);
        assert_eq!(10, batcher.comparisons());
    }

    #[test]
    fn test_sort() {
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![5, 1, 6, 2]));
            batcher.sort();
            assert_eq!(vec![1, 2, 5, 6], batcher.inner());
        }
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![5, 4, 3, 2, 1]));
            batcher.sort();
            assert_eq!(vec![1, 2, 3, 4, 5], batcher.inner());
        }
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![7, 6, 5, 4, 3, 2, 1]));
            batcher.sort();
            assert_eq!(vec![1, 2, 3, 4, 5, 6, 7], batcher.inner());
        }
        {
            let mut batcher = BatcherSort::new(ClearCmp::boxed(vec![1, 5, 6, 7, 2, 4, 3, 5]));
            batcher.sort();
            assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.inner());
        }
        {
            let k = 1;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![5, 1, 6, 7]), k);
            batcher.sort();
            assert_eq!(vec![1], batcher.vs.split_at(k).0);
        }
        {
            let k = 2;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; 4]), k);
            batcher.sort();
            assert_eq!(5, batcher.comparisons());
        }
        {
            let k = 2;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; 8]), k);
            batcher.sort();
            assert_eq!(13, batcher.comparisons());
        }
        {
            let k = 2;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; 16]), k);
            batcher.sort();
            assert_eq!(29, batcher.comparisons());
        }
        {
            let k = 3;
            let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; 10]), k);
            batcher.sort();
            assert_eq!(20, batcher.comparisons());
        }
    }

    #[quickcheck]
    fn prop_sort(xs: Vec<usize>) -> TestResult {
        if xs.len() > 20 {
            return TestResult::discard();
        }
        let mut sorted = xs.clone();
        sorted.sort();

        let mut batcher = BatcherSort::new(ClearCmp::boxed(xs));
        batcher.sort();
        TestResult::from_bool(batcher.inner() == sorted)
    }

    #[quickcheck]
    fn prop_sort_k(xs: Vec<u16>, k: usize) -> TestResult {
        if xs.len() > 20 {
            return TestResult::discard();
        }

        if k > xs.len() {
            return TestResult::discard();
        }

        let mut sorted = xs.clone();
        sorted.sort();

        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(xs), k);
        batcher.sort();

        TestResult::from_bool(batcher.vs.split_at(k).0 == sorted.split_at(k).0)
    }

    #[quickcheck]
    fn prop_sort_k_5(xs: Vec<u16>) -> TestResult {
        if xs.len() > 5000 || xs.len() < 1 {
            return TestResult::discard();
        }
        let k = 5usize.min(xs.len());

        let mut sorted = xs.clone();
        sorted.sort();

        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(xs), k);
        batcher.sort();

        TestResult::from_bool(batcher.vs.split_at(k).0 == sorted.split_at(k).0)
    }

    #[quickcheck]
    fn prop_sort_k_2(xs: Vec<u16>) -> TestResult {
        if xs.len() > 5000 || xs.len() < 1 {
            return TestResult::discard();
        }
        let k = 2usize.min(xs.len());

        let mut sorted = xs.clone();
        sorted.sort();

        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(xs), k);
        batcher.sort();

        TestResult::from_bool(batcher.vs.split_at(k).0 == sorted.split_at(k).0)
    }
}
