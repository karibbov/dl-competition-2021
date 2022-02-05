import os
from src.worker import PyTorchWorker
from src.utils import save_result
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB


def run_bohb(
        host: str,
        port: int,
        run_id: str,
        n_bohb_iterations: int,
        working_dir: str,
        min_budget: int,
        max_budget: int,
        n_min_workers: int):
    """Run BOHB.

    Returns:
        None

    """
    try:
        # Start a nameserver #####
        ns = hpns.NameServer(run_id=run_id, host=host, port=port,
                             working_directory=working_dir)
        ns_host, ns_port = ns.start()

        # Start local worker
        for i in range(n_min_workers):
            w = PyTorchWorker(run_id=run_id, host=host, nameserver=ns_host,
                                      nameserver_port=ns_port, id=i)
            w.run(background=True)
        # w = PyTorchWorker(run_id=run_id, host=host, nameserver=ns_host,
        #                   nameserver_port=ns_port)
        # w.run(background=True)

        # Run an optimizer
        bohb = BOHB(configspace=w.get_configspace(),
                    run_id=run_id,
                    host=host,
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    min_budget=min_budget, max_budget=max_budget)
        print(f"BUDGETS: {bohb.budgets}")
        result = bohb.run(n_iterations=n_bohb_iterations, min_n_workers=4)
        save_result('bohb_result', result)
    finally:
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()

    return result


if __name__ == '__main__':
    # minimum budget that BOHB uses
    min_budget = 1
    # largest budget BOHB will use
    max_budget = 2
    working_dir = os.curdir
    host = "localhost"
    port = 0
    run_id = 'bohb_run_1'
    n_bohb_iterations = 4
    res = run_bohb(
            host,
            port,
            run_id,
            n_bohb_iterations,
            working_dir,
            min_budget,
            max_budget,
            4)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    all_runs = res.get_all_runs()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations were sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
    print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

