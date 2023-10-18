use crate::AsyncComparator;
use rayon;
use std::sync::mpsc;
use std::thread;

#[derive(Copy, Clone, Debug)]
struct Task {
    v0: usize,
    v1: usize,
    level: usize,
}

#[derive(Copy, Clone, Debug)]
enum TaskState {
    Ok(Task),
    Blocked,
    Done,
}

struct TaskManager {
    remaining: Vec<Task>,
    processing: Vec<Task>,
}

impl TaskManager {
    fn new() -> Self {
        unimplemented!()
    }

    /// Output the next task in the queue.
    fn next_task(&mut self, finished_task: Task) -> TaskState {
        unimplemented!()
    }

    /// Output the initial tasks,
    /// i.e., tasks that have no dependency at level 0
    fn initial_tasks(&mut self) -> Vec<Task> {
        unimplemented!()
    }
}

fn do_work<CMP>(n_threads: usize, cmp: CMP, vs: &[CMP::Item])
where
    CMP: AsyncComparator + Sync + Send + Clone,
{
    let (pool_tx, pool_rx): (mpsc::Sender<Option<Task>>, mpsc::Receiver<Option<Task>>) =
        mpsc::channel();
    let (man_tx, man_rx): (mpsc::Sender<Task>, mpsc::Receiver<Task>) = mpsc::channel();

    let mut man = TaskManager::new();

    // start a thread for manager
    let man_handler = thread::spawn(move || {
        // send the initial tasks
        for task in man.initial_tasks() {
            pool_tx.send(Some(task)).unwrap();
        }

        loop {
            let finished_task = man_rx.recv().unwrap();
            match man.next_task(finished_task) {
                TaskState::Ok(task) => {
                    pool_tx.send(Some(task)).unwrap();
                }
                TaskState::Blocked => {
                    // Do nothing.
                    // Wait for the next finished_task to come in
                    // and hopefully it'll be unblocked.
                }
                TaskState::Done => {
                    pool_tx.send(None).unwrap();
                    return;
                }
            }
        }
    });

    // start the thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();
    pool.install(move || {
        loop {
            match pool_rx.recv().unwrap() {
                None => break,
                Some(task) => {
                    // perform the task in the thread pool
                    let man_tx = man_tx.clone();
                    let cmp = cmp.clone();
                    rayon::scope(move |s| {
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
    });

    man_handler.join().unwrap();
    // pool.join();
}

#[cfg(test)]
mod test {}
