import numpy as np
import bw2calc as bc
import logging
import pandas as pd
import brightway2 as bw

_log = logging.getLogger(__name__)

# From bw2calc
try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve


class MyMonteCarloLCA(bc.MonteCarloLCA):
    """Smarter iterative solution when doing contribution analysis.

    The original MonteCarloLCA stores only one `self.guess`, so if switching
    between calculating for multiple activities (like in contribution analysis)
    it does not solve efficiiently.

    Here we store different guesses for different demand vectors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cache guesses by current demand vector
        self.guesses = {}

    def new_sample(self):
        """Get new samples like __next__ but don't calculate anything."""
        if not hasattr(self, "tech_rng"):
            self.load_data()
        self.rebuild_technosphere_matrix(self.tech_rng.next())
        self.rebuild_biosphere_matrix(self.bio_rng.next())
        if self.lcia:
            self.rebuild_characterization_matrix(self.cf_rng.next())
        if self.weighting:
            self.weighting_value = self.weighting_rng.next()
        if self.presamples:
            self.presamples.update_matrices()

    def solve_linear_system(self):
        demand_sig = tuple(self.demand.keys())
        _log.debug("    Solve linear system: %s", demand_sig)
        guess = self.guesses.get(demand_sig)
        if not self.iter_solver or guess is None:
            _log.debug("      solving from scratch...")
            self.guesses[demand_sig] = guess = spsolve(
                self.technosphere_matrix, self.demand_array
            )
            _log.debug("      done")
            return guess
        else:
            _log.debug("      solving iteratively...")
            solution, status = self.iter_solver(
                self.technosphere_matrix,
                self.demand_array,
                x0=guess,
                atol="legacy",
                maxiter=1000,
            )
            _log.debug("      done (status %s)", status)
            if status != 0:
                _log.debug("      solving again from scratch...")
                self.guesses[demand_sig] = guess = spsolve(
                    self.technosphere_matrix, self.demand_array
                )
                _log.debug("      done")
                return guess
            else:
                return solution
            
    
    def get_activity_indices(self, act_name):
        """Return a dictionary with the indices of all products that come from the specified database."""
        indices = {}
        label = 'Foreground'
        
        # Find activity keys that come from the specified database
        for activity_key, activity_value in self.activity_dict.items():
            if activity_key[1] == act_name:
                activity_index = self.activity_dict.get(activity_key)
                if activity_index is not None:
                    # Store the activity index in the list corresponding to the label
                    if label not in indices:
                        indices[label] = []
                    indices[label].append(activity_index)
        
        # Sort the indices within the list
        indices[label].sort()
        
        return indices
    


def recursive_calculation(
    activity,
    final_activities,
    lcia_method,
    lca_obj=None,
    total_score=None,
    amount=1,
    level=0,
    max_level=3,
    cutoff=1e-2,
):
    """Contribution analysis back through supply chain to `final_activities`.

    `activity` is the starting activity.

    `final_activities` should be a dictionary of {activity: label}.

    `lcia_method` is the LCA impact assessment method as expected by Brightway.

    `max_level` and `cutoff` determine when the analysis gives up.

    Returns dictionary {label1: score1, label2: score2, ...}. Include an
    additional label "OTHER" if the activities in `final_activities` do not
    account for the full amount of the initial total score.

    Adapted from bw2analyzer.utils.print_recursive_calculation.

    """
    if lca_obj is None:
        _log.debug("  [initialising]")
        lca_obj = bc.LCA({activity: amount}, lcia_method)
        lca_obj.lci(factorize=True)
        lca_obj.lcia()
        total_score = lca_obj.score
        _log.debug("                 total score = %.2g", total_score)
    elif total_score is None:
        raise ValueError("Need total score and lca_obj")
    else:
        _log.debug("  [redo lcia] %s", activity)
        lca_obj.redo_lcia({activity: amount})
        if abs(lca_obj.score) <= abs(total_score * cutoff):
            return {}

    print("{:4.3f}{}{:4.3f} ({:06.4f}): {:.70}".format(amount,"  " * level, lca_obj.score / total_score, lca_obj.score, str(activity)))
    print("{}                {}".format("  " * level, str(activity.key)))
    if activity in final_activities:
        label = final_activities[activity]
        _log.debug("  [found] %s -> %s", activity, label)
        print("{}  --> {}".format("  " * level, str(label)))
        return {label: lca_obj.score}

    # Otherwise, traverse further
    result = {}
    if level < max_level:
        for exc in activity.technosphere():
            new_result = recursive_calculation(
                activity=exc.input,
                final_activities=final_activities,
                lcia_method=lcia_method,
                lca_obj=lca_obj,
                total_score=total_score,
                amount=amount*exc["amount"],
                level=level + 1,
                max_level=max_level,
                cutoff=cutoff,
            )
           
            for k, v in new_result.items():
                result[k] = result.get(k, 0) + v
    else:
        print("...max level...")


    # If top level, calculate "other"
    if level == 0:
        #print(total_score)
        num_expected_labels = len(set(final_activities.values()))
        if len(result) != num_expected_labels:
            _log.warning(
                "Warning: only %d out of %d found", len(result), num_expected_labels
            )
        total_accounted_for = sum(result.values())
        missing = total_score - total_accounted_for
        #print(missing)
        assert missing / total_score > -1e6, "missing should be nearly positive"
        result["Total"] = total_score
        if missing > 1e-6 * total_score:
            result["OTHER"] = missing
        

    return result


def sample_comparative_contribution(lca, demands, final_activities, **kwargs):
    """Draw a sample from `lca` and do contribution analysis.

    `lca` must be an instance of `MyMonteCarlo`, already prepared for LCIA
    calculations. Each time this function is called, the technosphere matrices
    are updated with a new Monte Carlo sample, and the LCIA calculations are
    repeated for each of the demands in `demands`. For each,
    `recursive_calculation` is called to calculate the contribution analysis
    back to the activities in `final_activities`.

    """
    # Update matrices from random number generator
    _log.debug("New sample...")
    lca.new_sample()
    _log.debug("done")

    # Do the calculation for each demand vector
    results = []
    for demand in demands:
        # This is not ideal, computationally, since we are using an
        # iterative solver for the MC samples, then factorizing anyway
        # for the contribution analysis...

        if lca is None:
            _log.debug("  [initialising]")
            lca = bc.LCA({activity: amount}, lcia_method)
            lca.lci(factorize=True)
            lca.lcia()
            total_score = lca.score
        else:
            lca.decompose_technosphere()

            _log.debug("Contributions to %s", demand)
            lca.redo_lci(demand)
            lca.redo_lcia(demand)

            grouper = ScoreGrouper(lca)
            contributions = grouper(final_activities)
            results.append(contributions)

    return results


import numpy as np
import pandas as pd

def collect_contribution_samples(
    lca,
    demands,
    final_activities,
    num_samples,
    method_label,
    demand_labels,
    component_order,
    activity_labels=None,  # Set default value to None
    **kwargs
    ):
    """Repeatedly call `sample_comparative_contribution` and collect results in
    a DataFrame, including quantities of specified processes."""
    samples = []
    
    for iteration in range(num_samples):
        results = sample_comparative_contribution(
            lca, demands, final_activities, **kwargs
        )
        
        if activity_labels:
            grouper = ScoreGrouper(lca)
            group_indices = grouper.get_group_indices(activity_labels)
            # Call solve_supply_subset for each group of indices
            activity_supplies = {label: grouper.solve_supply_subset_test(indices) for label, indices in group_indices.items()}
        else:
            activity_supplies = {}

        for label, result in zip(demand_labels, results):
            order = list(component_order) + [
                k for k in result if k not in component_order
            ]

            for k in order:
                row = [label, k, method_label, iteration, result.get(k, 0)]
                
                if activity_labels:
                    # Sum the supply values for all activity labels
                    total_supply_value = sum(np.sum(supply_values) for supply_values in activity_supplies.values())
                    row.append(total_supply_value)
                else:
                    row.append(None)  # Handle the case where activity_labels is not provided
                
                samples.append(row)

    # Define columns
    columns = ["scenario", "component", "method", "iteration", "score", "activity labels"]
    
    # Create DataFrame
    return pd.DataFrame(samples, columns=columns)




class ScoreGrouper:
    """Allocate LCIA score to groups of processes."""
    
    def __init__(self, lca_obj):
        self.lca_obj = lca_obj

        # First get the score per process activity, which is constant and
        # can be cached
        self.characterized_biosphere = np.array(
            (lca_obj.characterization_matrix * lca_obj.biosphere_matrix)
            .sum(axis=0)
        ).ravel()
        
        # Track the part of the supply (process activity) which has been
        # "used" so far
        self.used_supply = np.zeros_like(lca_obj.supply_array)
        self.used_indices = set()
        
    def get_group_indices(self, activity_labels):
        """Return {label: indices} from input {activity_key: label}"""
        indices = {}
        seen_keys = {}
        for key, label in activity_labels.items():
            if key in seen_keys:
                raise ValueError(f"Key {key} ({label}) already mapped to {seen_keys[key]}")
            seen_keys[key] = label
            if label not in indices:
                indices[label] = []
            if key in self.lca_obj.activity_dict:
                index = self.lca_obj.activity_dict[key]
                indices[label].append(index)
        return indices

    def solve_supply_subset(self, indices):
        """Solve the process activity (supply) driven by only `indices`.
        
        This is a subset of the full LCI solution.
        """
        # The demand is based on supply and the technosphere coefficient
        demand = np.zeros(len(self.lca_obj.supply_array))
        demand[indices] = (
            self.lca_obj.supply_array[indices] *
            self.lca_obj.technosphere_matrix.diagonal()[indices]
        )
        
        # Solve for this subset of demand
        supply_subset = self.lca_obj.solver(demand)
        
        return supply_subset
    
    def solve_supply_subset_test(self, indices):
        """Solve the process activity (supply) driven by only `indices`.
        
        This is a subset of the full LCI solution.
        """
        # The demand is based on supply and the technosphere coefficient
        demand = np.zeros(len(self.lca_obj.supply_array))
        demand[indices] = (
            self.lca_obj.supply_array[indices] *
            self.lca_obj.technosphere_matrix.diagonal()[indices]
        )
        #print(self.lca_obj.supply_array[indices])
        
        # Solve for this subset of demand
        supply_subset_test = self.lca_obj.supply_array[indices]
        
        return supply_subset_test
    

    def calc_score(self, supply):
        """Calculate LCIA score for solution `supply`."""
        score = float((self.characterized_biosphere * supply).sum())
        return score
        
    def get_cumulative_score(self, indices):
        """Return cumulative LCA impact score based on only `indices`.

        First, look up how much activity corresponding to the processes with indices `indices`
        is occuring in the current solution.  Then, re-evaluate what the impact of the system
        with that subset of demand would be.

        Partly based on the GraphTraversal implementation.
        """
        indices_already_used = self.used_indices & set(indices)
        if indices_already_used:
            raise ValueError(f"Already got score for processes: {indices_already_used}")
        self.used_indices.update(indices)
        
        # Solve for this subset of demand
        supply_subset = self.solve_supply_subset(indices)
        self.used_supply += supply_subset
        
        # Had considered checking for this supply exceeding the original and clipping or
        # issuing a warning, but it's ok to temporarily exceed as later contributions
        # might reduce it again.  Instead, we can check at the end.        
        
        # Cumulative score
        score = self.calc_score(supply_subset)
        return score
    
    def residual_score(self):
        """Calculate residual LCIA score for processes which have not yet been reported."""
        # Solve for remaining activities which have not been reported separately
        residual_supply = self.lca_obj.supply_array - self.used_supply
        return self.calc_score(residual_supply)
    
        
            
    def residual_activities(self, include_databases=None, exclude_databases=None):
        """Report which activities are contributing to the residual score.
        
        If `include_databases` is given, only include activities from those databases.
        Activities from any databases in `exclude_databases` are ignored.
        
        Example::
        
            residual = grouper.residual_score()
            for key, score in grouper.residual_activities(exclude_databases={"cutoff38"}):
                print(f"{score/residual:5.0%}  {bd.get_activity(key)}")
        """
        # Look up activities from indices
        adict, _, _ = self.lca_obj.reverse_dict()
        
        if exclude_databases is None:
            exclude_databases = set()
        all_indices = (i for i in range(len(self.lca_obj.supply_array)))
        if include_databases is not None:
            foreground_indices = [i for i in all_indices
                                  if adict[i][0] in include_databases and adict[i][0] not in exclude_databases]
        else:
            foreground_indices = [i for i in all_indices
                                  if adict[i][0] not in exclude_databases]
        
        results = []
        for i in foreground_indices:
            solution = self.solve_supply_subset([i])
            # Clip solution to only include parts that we haven't reported already
            excess = np.maximum(0, (self.used_supply + solution) - self.lca_obj.supply_array)
            solution -= excess
            score = self.calc_score(solution)
            if abs(score) > 1e-4:
                results.append((adict[i], score))

        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results
    
   
        
            
    def __call__(self, activity_labels):
        group_indices = self.get_group_indices(activity_labels)
        scores = {
            label: self.get_cumulative_score(indices)
            for label, indices in group_indices.items()
        }
        #residual = self.residual_score()
        #if abs(residual) > abs(np.array(list(scores.values()))).max() * 1e-3:
        #    scores["OTHER"] = residual 
        return scores
