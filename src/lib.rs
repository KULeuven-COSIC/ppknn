pub struct BatcherSort<T>
where
    T: Ord,
{
    pub vs: Vec<T>,
}

impl<T> BatcherSort<T>
where
    T: Ord,
{
    pub fn new(vs: Vec<T>) -> Self {
        Self { vs }
    }

    pub fn merge(&mut self, lo: usize, n: usize, r: usize) {
        let m = r * 2;
        if m < n {
            self.merge(lo, n, m);
            self.merge(lo + r, n, m);
            for i in (lo + r..lo + n - r).step_by(m) {
                compare_at(&mut self.vs, i, i + r);
            }
        } else {
            compare_at(&mut self.vs, lo, lo + r);
        }
    }
}

fn compare_at<T>(vs: &mut Vec<T>, i: usize, j: usize)
where
    T: Ord,
{
    if vs[i] > vs[j] {
        vs.swap(i, j);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_merge_2() {
        let mut batcher = BatcherSort::new(vec![2, 1]);
        batcher.merge(0, 2, 1);
        assert_eq!(vec![1, 2], batcher.vs);
    }

    #[test]
    fn test_merge_4() {
        let mut batcher = BatcherSort::new(vec![1, 5, 2, 4]);
        batcher.merge(0, 4, 1);
        assert_eq!(vec![1, 2, 4, 5], batcher.vs);
    }

    #[test]
    fn test_merge_8() {
        let mut batcher = BatcherSort::new(vec![1, 5, 6, 7, 2, 3, 4, 5]);
        batcher.merge(0, 8, 1);
        assert_eq!(vec![1, 2, 3, 4, 5, 5, 6, 7], batcher.vs);
    }
}
