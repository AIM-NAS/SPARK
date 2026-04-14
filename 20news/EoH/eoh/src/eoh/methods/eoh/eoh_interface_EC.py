import numpy as np
import time
import warnings
import re
import concurrent.futures

from joblib import Parallel, delayed

from .eoh_evolution import Evolution
from .evaluator_accelerate import add_numba_decorator


class InterfaceEC():
    def __init__(
        self,
        pop_size,
        m,
        api_endpoint,
        api_key,
        llm_model,
        llm_use_local,
        llm_local_url,
        debug_mode,
        interface_prob,
        select,
        n_p,
        timeout,
        use_numba,
        **kwargs
    ):
        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(
            api_endpoint,
            api_key,
            llm_model,
            llm_use_local,
            llm_local_url,
            debug_mode,
            prompts,
            **kwargs
        )
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba

    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            file.write(code)
        return

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate(self, population, code):
        if code is None:
            return False
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def _make_invalid_offspring(self, error_msg=None):
        return {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': {'error': error_msg} if error_msg is not None else None
        }

    def _safe_parent_selection(self, pop, n_parent, operator):
        """
        Safely select parents. Return None if population is insufficient.
        """
        pop = [] if pop is None else pop

        if len(pop) < n_parent:
            msg = f"skip {operator}: need >= {n_parent} parent(s), got {len(pop)}"
            if self.debug:
                print(f"[InterfaceEC] {msg}")
            return None

        parents = self.select.parent_selection(pop, n_parent)

        if parents is None:
            msg = f"skip {operator}: parent_selection returned None"
            if self.debug:
                print(f"[InterfaceEC] {msg}")
            return None

        if len(parents) < n_parent:
            msg = f"skip {operator}: parent_selection returned {len(parents)} parent(s), need {n_parent}"
            if self.debug:
                print(f"[InterfaceEC] {msg}")
            return None

        return parents

    def population_generation(self):
        n_create = 2
        population = []

        for _ in range(n_create):
            _, pop = self.get_algorithm([], 'i1')
            for p in pop:
                population.append(p)

        return population

    def population_generation_seed(self, seeds, n_p):
        population = []

        fitness = Parallel(n_jobs=n_p)(
            delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds
        )

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")
        return population

    def _get_alg(self, pop, operator):
        """
        Return:
            parents, offspring(dict)
        offspring structure:
            {
                'algorithm': ...,
                'code': ...,
                'objective': None,
                'other_inf': ...
            }
        """
        offspring = self._make_invalid_offspring()
        parents = None

        if operator == "i1":
            parents = None
            code, algorithm = self.evol.i1()
            offspring['code'] = code
            offspring['algorithm'] = algorithm
            return parents, offspring

        elif operator == "e1":
            parents = self._safe_parent_selection(pop, self.m, operator)
            if parents is None:
                return None, self._make_invalid_offspring(f"insufficient parents for {operator}")
            code, algorithm = self.evol.e1(parents)
            offspring['code'] = code
            offspring['algorithm'] = algorithm
            return parents, offspring

        elif operator == "e2":
            parents = self._safe_parent_selection(pop, self.m, operator)
            if parents is None:
                return None, self._make_invalid_offspring(f"insufficient parents for {operator}")
            code, algorithm = self.evol.e2(parents)
            offspring['code'] = code
            offspring['algorithm'] = algorithm
            return parents, offspring

        elif operator == "m1":
            parents = self._safe_parent_selection(pop, 1, operator)
            if parents is None:
                return None, self._make_invalid_offspring(f"insufficient parents for {operator}")
            code, algorithm = self.evol.m1(parents[0])
            offspring['code'] = code
            offspring['algorithm'] = algorithm
            return parents, offspring

        elif operator == "m2":
            parents = self._safe_parent_selection(pop, 1, operator)
            if parents is None:
                return None, self._make_invalid_offspring(f"insufficient parents for {operator}")
            code, algorithm = self.evol.m2(parents[0])
            offspring['code'] = code
            offspring['algorithm'] = algorithm
            return parents, offspring

        elif operator == "m3":
            parents = self._safe_parent_selection(pop, 1, operator)
            if parents is None:
                return None, self._make_invalid_offspring(f"insufficient parents for {operator}")
            code, algorithm = self.evol.m3(parents[0])
            offspring['code'] = code
            offspring['algorithm'] = algorithm
            return parents, offspring

        else:
            msg = f"Evolution operator [{operator}] has not been implemented!"
            if self.debug:
                print(msg)
            return None, self._make_invalid_offspring(msg)

    def _maybe_add_numba(self, code):
        if not self.use_numba:
            return code

        if code is None:
            raise ValueError("offspring code is None when use_numba=True")

        pattern = r"def\s+(\w+)\s*\(.*\):"
        match = re.search(pattern, code)

        if match is None:
            raise ValueError("No function definition found for numba decoration.")

        function_name = match.group(1)
        return add_numba_decorator(program=code, function_name=function_name)

    def get_offspring(self, pop, operator):
        try:
            p, offspring = self._get_alg(pop, operator)

            if offspring is None:
                return None, self._make_invalid_offspring(f"{operator}: offspring is None")

            if offspring.get('code') is None:
                if self.debug:
                    print(f"[InterfaceEC] skip {operator}: offspring code is None")
                return p, offspring

            code = self._maybe_add_numba(offspring['code'])

            n_retry = 1
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")

                p, offspring = self._get_alg(pop, operator)

                if offspring is None or offspring.get('code') is None:
                    return p, self._make_invalid_offspring(f"{operator}: retry offspring invalid")

                code = self._maybe_add_numba(offspring['code'])

                if n_retry > 1:
                    break

            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     future = executor.submit(self.interface_eval.evaluate, code)
            #     fitness = future.result()
            #     offspring['objective'] = np.round(fitness, 5)
            fitness = self.interface_eval.evaluate(code)
            offspring['objective'] = np.round(fitness, 5)

        except Exception as e:
            if self.debug:
                print(f"[InterfaceEC] get_offspring exception: {repr(e)}")
            offspring = self._make_invalid_offspring(repr(e))
            p = None

        return p, offspring

    def get_algorithm(self, pop, operator):
        results = []
        try:
            results = Parallel(n_jobs=self.n_p)(
                delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size)
            )
        except Exception as e:
            if self.debug:
                print(f"Error: {repr(e)}")
            print("Parallel time out .")

        time.sleep(2)

        out_p = []
        out_off = []

        for p, off in results:
            if off is None:
                continue

            if off.get('code') is None or off.get('objective') is None:
                if self.debug:
                    print(f">>> skip invalid offspring: {off}")
                continue

            out_p.append(p)
            out_off.append(off)

            if self.debug:
                print(f">>> check offsprings: \n {off}")

        return out_p, out_off