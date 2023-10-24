use crate::AsyncComparator;
use rayon;
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Task {
    v0: usize,
    v1: usize,
    level: usize,
}

impl Task {
    pub fn new(v0: usize, v1: usize, level: usize) -> Self {
        Self { v0, v1, level }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TaskState {
    Ok(Task),
    Blocked,
    Done,
}

struct TaskManager {
    // TODO consider using linked list since we're doing a lot of removes
    remaining: Vec<Task>,
    processing: Vec<Task>,
}

impl TaskManager {
    fn new(network: &[Task]) -> Self {
        Self {
            remaining: network.to_vec(),
            processing: vec![],
        }
    }

    // we need to be careful for situations such as
    // processing: (0, 1)
    // remaining: (1, 2) (2, 3)
    // since we skip (1, 2) as it conflicts with (0, 1)
    // but we should not accept (2, 3) since it depends on (1, 2)
    // To avoid this, we will only accept tasks that are 1 level higher
    // than the tasks in `self.processing`
    fn conflict_with_processing(&self, task: &Task) -> bool {
        if self.processing.is_empty() {
            return false;
        }

        let mut max_level = 0;
        for t in self.processing.iter() {
            if max_level < t.level {
                max_level = t.level;
            }

            // tasks on the same level should never conflict
            if task.level == t.level {
                continue;
            }

            // otherwise, the task at a higher level
            // may depend on tasks on lower levels,
            // not necessarily one level lower
            if task.v0 == t.v0 || task.v1 == t.v0 || task.v0 == t.v1 || task.v1 == t.v1 {
                return true;
            }
        }

        // the new task cannot be 2 or more levels higher than
        // the tasks in `self.processing` since we do not allow
        // our algorithm to "skip" levels, all the tasks are sorted by level
        if max_level == task.level || max_level + 1 == task.level {
            false
        } else {
            true
        }
    }

    fn remove_finished_task(&mut self, finished_task: Task) {
        let index = self
            .processing
            .iter()
            .position(|x| *x == finished_task)
            .unwrap();
        self.processing.remove(index);
    }

    /// Output the next task in the queue.
    /// There are a few steps for finding the next task when one is finished
    /// - take `finished_task` out of `self.processing`
    /// - find the next task that does not "conflict" with any current tasks
    fn next_task(&mut self, finished_task: Task) -> TaskState {
        self.remove_finished_task(finished_task);

        if self.remaining.is_empty() {
            return TaskState::Done;
        }

        // Iterate over the remaining tasks on the same level and see
        // if there is one we can execute.
        // They must be on the same level because only this
        // way we can guarantee that the execution order for these tasks
        // do not matter.
        let curr_level = self.remaining[0].level;
        for i in 0..self.remaining.len() {
            if self.remaining[i].level == curr_level
                && !self.conflict_with_processing(&self.remaining[i])
            {
                let t = self.remaining.remove(i);
                self.processing.push(t);
                return TaskState::Ok(t);
            }
        }

        return TaskState::Blocked;
    }

    /// Output the initial tasks,
    /// i.e., tasks that have no dependency at level 0
    fn initial_tasks(&mut self, n_threads: usize) -> Vec<Task> {
        // we need at least one comparator at level 0
        assert_eq!(self.remaining[0].level, 0);
        // we must be at the start of the tasks
        assert!(self.processing.is_empty());

        let mut i = 0usize;
        loop {
            if self.remaining.is_empty() {
                break;
            }
            if i >= n_threads {
                break;
            }
            if self.remaining[i].level == 0 {
                i += 1;
            } else {
                break;
            }
        }
        self.processing = self.remaining.drain(0..i).collect();
        self.processing.clone()
    }
}

pub fn do_work<CMP>(n_threads: usize, network: &[Task], cmp: CMP, vs: &[CMP::Item])
where
    CMP: AsyncComparator + Sync + Send + Clone,
{
    let (pool_tx, pool_rx): (mpsc::Sender<Option<Task>>, mpsc::Receiver<Option<Task>>) =
        mpsc::channel();
    let (man_tx, man_rx): (mpsc::Sender<Task>, mpsc::Receiver<Task>) = mpsc::channel();

    let mut man = TaskManager::new(network);

    // start a thread for manager
    let man_handler = thread::spawn(move || {
        // send the initial tasks
        for task in man.initial_tasks(n_threads) {
            pool_tx.send(Some(task)).unwrap();
        }

        // even if there are more threads than initial tasks
        // we cannot send more since the initial tasks on level 0
        // always conflicts with tasks on level 1

        // listen to job completion and figure out the next tasks
        let mut done = false;
        loop {
            if done && man.processing.is_empty() {
                // println!("done");
                return;
            }

            let finished_task = man_rx.recv().unwrap();
            let next = man.next_task(finished_task);
            // println!("done: {:?}, next: {:?}", finished_task, next);
            match next {
                TaskState::Ok(task) => {
                    assert!(!done);
                    pool_tx.send(Some(task)).unwrap();
                }
                TaskState::Blocked => {
                    // Do nothing.
                    // Wait for the next finished_task to come in
                    // and hopefully it'll be unblocked.
                }
                TaskState::Done => {
                    if !done {
                        // we only need to send the None signal once
                        // for the threadpool to stop receiving tasks
                        pool_tx.send(None).unwrap();
                        done = true;
                    }
                }
            }
        }
    });

    // start the thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();
    loop {
        match pool_rx.recv().unwrap() {
            None => break,
            Some(task) => {
                // perform the task in the thread pool
                let man_tx = man_tx.clone();
                let cmp = cmp.clone();
                pool.scope(move |s| {
                    s.spawn(move |_| {
                        // do work
                        cmp.compare(&vs[task.v0], &vs[task.v1]);
                        // send the manager a message when the task is done
                        man_tx.send(task).unwrap();
                    });
                });
            }
        }
    }
    man_handler.join().unwrap();
}

pub fn load_network(path: &Path) -> std::io::Result<Vec<Task>> {
    let mut out: Vec<Task> = vec![];
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .trim(csv::Trim::All)
        .from_path(path)?;

    // TODO figure out capacity
    let mut level_map: HashMap<usize, usize> = HashMap::new();
    for result in rdr.records() {
        let record = result?;
        // TODO better error handling
        let v0 = record.get(0).unwrap().parse::<usize>().unwrap();
        let v1 = record.get(1).unwrap().parse::<usize>().unwrap();

        // find the level for which a conflict exists
        let level_0 = level_map.get(&v0).map(|x| *x);
        let level_1 = level_map.get(&v1).map(|x| *x);
        let level = match (level_0, level_1) {
            (None, None) => {
                level_map.insert(v0, 0);
                level_map.insert(v1, 0);
                0
            }
            (None, Some(x)) => {
                level_map.insert(v0, x + 1);
                level_map.insert(v1, x + 1);
                x + 1
            }
            (Some(x), None) => {
                level_map.insert(v0, x + 1);
                level_map.insert(v1, x + 1);
                x + 1
            }
            (Some(x), Some(y)) => {
                let z = x.max(y);
                level_map.insert(v0, z + 1);
                level_map.insert(v1, z + 1);
                z + 1
            }
        };
        out.push(Task::new(v0, v1, level));
    }
    Ok(out)
}

#[cfg(test)]
mod test {
    use std::{
        path::PathBuf,
        sync::{Arc, Mutex},
    };

    use rand::Rng;

    use crate::AsyncClearComparator;

    use super::*;

    #[test]
    fn test_task_manager_basic() {
        let network = vec![Task::new(0, 1, 0), Task::new(1, 2, 1)];
        let mut man = TaskManager::new(&network);

        // check the initial task
        // even if there are 2 threads, only spawn one since the rest have higher level
        assert_eq!(man.initial_tasks(2), vec![Task::new(0, 1, 0)]);

        // now we have one tasks in processing and it conflicts
        assert!(man.conflict_with_processing(&Task::new(1, 2, 1)));

        // suppose the next task is finished,
        // we should obtain a new task
        assert_eq!(
            TaskState::Ok(Task::new(1, 2, 1)),
            man.next_task(Task::new(0, 1, 0))
        );

        // finally we should receive done after the last task is completed
        assert_eq!(TaskState::Done, man.next_task(Task::new(1, 2, 1)));
        assert!(man.processing.is_empty());
        assert!(man.remaining.is_empty());
    }

    #[test]
    fn test_thread_pool_basic() {
        let cmp = AsyncClearComparator::new();
        let network = vec![Task::new(0, 1, 0), Task::new(1, 2, 1)];
        let actual = vec![5, 1, 6];

        let a_actual: Vec<_> = actual.iter().map(|x| Arc::new(Mutex::new(*x))).collect();
        do_work(2, &network, cmp, &a_actual);
        let a_actual: Vec<_> = a_actual.into_iter().map(|x| *x.lock().unwrap()).collect();
        assert_eq!(vec![1, 5, 6], a_actual);
    }

    #[test]
    fn test_load_network() {
        let d: PathBuf = [env!("CARGO_MANIFEST_DIR"), "data"].iter().collect();
        {
            let mut d = d.clone();
            d.push("test_network1.csv");
            let network = load_network(d.as_path()).unwrap();
            assert_eq!(network.len(), 4);
            assert_eq!(network[0], Task::new(0, 1, 0));
            assert_eq!(network[1], Task::new(1, 2, 1));
            assert_eq!(network[2], Task::new(2, 3, 2));
            assert_eq!(network[3], Task::new(3, 4, 3));
        }
        {
            let mut d = d.clone();
            d.push("test_network2.csv");
            let network = load_network(d.as_path()).unwrap();
            assert_eq!(network.len(), 4);
            assert_eq!(network[0], Task::new(0, 1, 0));
            assert_eq!(network[1], Task::new(2, 3, 0));
            assert_eq!(network[2], Task::new(4, 5, 0));
            assert_eq!(network[3], Task::new(6, 7, 0));
        }
        {
            let mut d = d.clone();
            d.push("test_network3.csv");
            let network = load_network(d.as_path()).unwrap();
            assert_eq!(network.len(), 4);
            assert_eq!(network[0], Task::new(0, 1, 0));
            assert_eq!(network[1], Task::new(1, 2, 1));
            assert_eq!(network[2], Task::new(2, 3, 2));
            assert_eq!(network[3], Task::new(0, 4, 1));
        }
    }

    fn test_network(d: usize, k: usize, n_threads: usize) {
        let pb: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "data",
            &format!("network-{}-{}.csv", d, k),
        ]
        .iter()
        .collect();
        let network = load_network(pb.as_path()).unwrap();

        let cmp = AsyncClearComparator::new();
        let mut rng = rand::thread_rng();
        let actual: Vec<_> = (0..d).map(|_| rng.gen::<u64>()).collect();
        let a_actual: Vec<_> = actual.iter().map(|x| Arc::new(Mutex::new(*x))).collect();
        do_work(n_threads, &network, cmp, &a_actual);
        let a_actual: Vec<_> = a_actual.into_iter().map(|x| *x.lock().unwrap()).collect();

        // find the largest of the first k values
        // then check that everything else is larger or equal
        let (left, right) = a_actual.split_at(k);
        let left_max = left.iter().max().unwrap();
        assert!(right.iter().all(|r| { r >= left_max }));
    }

    #[test]
    fn test_network_100_5() {
        test_network(100, 5, 2)
    }

    #[test]
    fn test_network_1000_50() {
        test_network(1000, 50, 2)
    }
}
