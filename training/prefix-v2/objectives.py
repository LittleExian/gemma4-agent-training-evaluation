"""Explicit training objectives. No torch dependency; testable on CPU.

Task-mean averages tokens within a decision, decisions within a task, then
averages tasks. Uniform row sampling uses inverse task-size importance weights.
Batch denominator is the number of sampled rows, NOT the sampled weights sum;
self-normalizing each batch would change the intended objective.
"""
import collections
import math

def task_scales(rows):
    counts = collections.Counter(row['task_id'] for row in rows)
    if not rows: raise ValueError('Empty training data')
    return {task: len(rows)/(len(counts)*count) for task,count in counts.items()}

def validate_rows(rows):
    seen=set()
    for row in rows:
        if row['sample_id'] in seen: raise ValueError('Duplicate sample id')
        seen.add(row['sample_id'])
        if len(row['input_ids']) != len(row['weights']): raise ValueError('Mismatched tokens and weights')
        if any(not math.isfinite(w) or w<0 for w in row['weights']): raise ValueError('Invalid loss weight')
        if sum(row['weights'][1:])<=0: raise ValueError('No supervised tokens')

def scalar_loss(numerator, weight_sum, task_scale, objective):
    if objective == 'token_mean': return numerator
    if objective == 'task_mean': return numerator/weight_sum*task_scale
    raise ValueError(f'Unknown objective: {objective}')
