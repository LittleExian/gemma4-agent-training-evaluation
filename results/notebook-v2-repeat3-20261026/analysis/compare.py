"""Compare fixed-seed Notebook v2 outcomes; public scalar fields only."""
import collections
import csv
import itertools
import json
from pathlib import Path
import statistics

ROUNDS = (1, 2, 3)
HARDWARE = {
    1: 'GPU3: 2 workers; interrupted and resumed across two cold model services',
    2: 'GPU3: 2 workers',
    3: 'GPU2+GPU3: 2 workers per card, 4 total',
}


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def analyze(source):
    grouped = collections.defaultdict(dict)
    for row in source:
        identifier, number = row['instance_id'], row['round']
        if number not in ROUNDS or number in grouped[identifier]:
            raise ValueError('Unexpected or duplicate task/round')
        grouped[identifier][number] = row
    tasks, eligible, excluded = [], {}, []
    for identifier, rounds in sorted(grouped.items()):
        if set(rounds) != set(ROUNDS):
            raise ValueError('All three rounds are required for every task')
        flags = [rounds[n]['status'] == 'preflight_excluded' for n in ROUNDS]
        if len(set(flags)) != 1:
            raise ValueError('Eligibility changed between repetitions')
        if any(rounds[n]['status'] not in ('preflight_excluded','completed','agent_timeout') for n in ROUNDS):
            raise ValueError('Non-evaluation terminal state cannot be compared')
        if flags[0]:
            excluded.append(identifier)
            pattern, same = 'not_scored', None
        else:
            eligible[identifier] = rounds
            pattern = ''.join(str(int(rounds[n]['resolved'])) for n in ROUNDS)
            same = len({rounds[n]['patch_sha256'] for n in ROUNDS}) == 1
        tasks.append({'instance_id':identifier, 'repo':rounds[1]['repo'], 'pattern':pattern,
                      'correctness_changed':not flags[0] and len(set(pattern)) > 1,
                      'patches_all_equal':same, 'rounds':rounds})
    patterns = {p:{'count':0,'task_ids':[]} for p in ('000','001','010','011','100','101','110','111')}
    for task in tasks:
        if task['pattern'] != 'not_scored':
            patterns[task['pattern']]['task_ids'].append(task['instance_id'])
    for item in patterns.values():
        item['count'] = len(item['task_ids'])
    by_round=[]
    for n in ROUNDS:
        rows=[g[n] for g in eligible.values()]
        passed=sum(r['resolved'] is True for r in rows)
        by_round.append({'round':n,'total_tasks':len(grouped),'eligible_tasks':len(rows),'excluded_tasks':len(excluded),
                         'resolved':passed,'accuracy':passed/len(rows) if rows else 0,
                         'timeouts':sum(r['agent_timed_out'] is True for r in rows),
                         'timeouts_that_passed':sum(r['agent_timed_out'] is True and r['resolved'] is True for r in rows),
                         'agent_seconds_mean':mean(r.get('agent_seconds') for r in rows),
                         'agent_seconds_median':median(r.get('agent_seconds') for r in rows),
                         'full_seconds_mean':mean(r.get('full_seconds') for r in rows),
                         'hardware':HARDWARE[n]})
    pairs=[]
    for a,b in itertools.combinations(ROUNDS,2):
        aset={i for i,v in eligible.items() if v[a]['resolved']}
        bset={i for i,v in eligible.items() if v[b]['resolved']}
        pairs.append({'from_round':a,'to_round':b,'newly_passed_ids':sorted(bset-aset),
                      'lost_pass_ids':sorted(aset-bset),'unchanged_pass_count':len(aset&bset),
                      'unchanged_fail_count':len(set(eligible)-(aset|bset)),
                      'correctness_difference_count':len(aset^bset),
                      'pass_intersection_count':len(aset&bset),'pass_union_count':len(aset|bset),
                      'pass_jaccard':len(aset&bset)/len(aset|bset) if aset|bset else 1.0,
                      'raw_patch_difference_count':sum(v[a]['patch_sha256']!=v[b]['patch_sha256'] for v in eligible.values()),
                      'same_patch_different_correctness_ids':sorted(i for i,v in eligible.items()
                          if v[a]['patch_sha256']==v[b]['patch_sha256'] and v[a]['resolved']!=v[b]['resolved'])})
    repos=[]
    for repo in sorted({v[1]['repo'] for v in eligible.values()}):
        rows={i:v for i,v in eligible.items() if v[1]['repo']==repo}
        repos.append({'repo':repo,'eligible_tasks':len(rows),
                      'resolved_by_round':{str(n):sum(v[n]['resolved'] for v in rows.values()) for n in ROUNDS},
                      'variable_correctness_count':sum(len({v[n]['resolved'] for n in ROUNDS})>1 for v in rows.values())})
    accuracies=[r['accuracy'] for r in by_round]
    result={'schema_version':1,'method':'notebook-v2','seed':42,'total_tasks':len(grouped),
            'eligible_tasks':len(eligible),'excluded_tasks':excluded,'rounds':by_round,
            'patterns':patterns,'all_passed_count':patterns['111']['count'],'all_failed_count':patterns['000']['count'],
            'variable_correctness_count':sum(t['correctness_changed'] for t in tasks),
            'passed_at_least_once_count':len(eligible)-patterns['000']['count'],
            'mean_accuracy':mean(accuracies),'accuracy_spread_percentage_points':100*(max(accuracies)-min(accuracies)),
            'patches_identical_all_rounds':sum(t['patches_all_equal'] is True for t in tasks),
            'identical_empty_patch_ids':[t['instance_id'] for t in tasks if t['patches_all_equal'] is True and t['rounds'][1]['patch_bytes']==0],
            'identical_nonempty_patch_ids':[t['instance_id'] for t in tasks if t['patches_all_equal'] is True and t['rounds'][1]['patch_bytes']>0],
            'pairwise':pairs,'by_repo':repos,
            'interpretation':'All three seeds are 42; R3 uses two GPUs/four workers versus one GPU/two workers in R1/R2. R1 was interrupted. Differences are observed repeatability, not a controlled estimate of random-seed variation or a causal hardware effect.'}
    return result,tasks


def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def flatten(task):
    out={k:task[k] for k in ('instance_id','repo','pattern','correctness_changed','patches_all_equal')}
    for n in ROUNDS:
        r=task['rounds'][n]
        out['r%d_correctness'%n]='not_scored' if task['pattern']=='not_scored' else 'passed' if r['resolved'] else 'failed'
        out['r%d_trace'%n]='round-%d/%s/%s.json'%(n,'eligibility' if task['pattern']=='not_scored' else 'traces',task['instance_id'])
        for key in ('resolved','status','agent_timed_out','agent_seconds','full_seconds','tool_calls','llm_calls','patch_sha256','patch_bytes'):
            out['r%d_%s'%(n,key)]=r.get(key)
    return out


def markdown(report,tasks):
    lines=['# Notebook v2 三轮结果与逐题差异','',
           '全部三轮 `seed=42`，每轮129题，其中126题可评；相同3题预先排除。通过与超时是独立字段，不能视为互斥。','',
           '| 轮次 | 通过 / 可评 | 正确率 | 超时 | 超时但通过 | Agent平均秒数 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in report['rounds']:
        lines.append('| %d | %d/%d | %.2f%% | %d | %d | %.2f |'%(r['round'],r['resolved'],r['eligible_tasks'],100*r['accuracy'],r['timeouts'],r['timeouts_that_passed'],r['agent_seconds_mean']))
    lines += ['', '三轮平均正确率 **%.2f%%**；最高与最低相差 **%.2f个百分点**。'%(100*report['mean_accuracy'],report['accuracy_spread_percentage_points']),
              '', '**%d题三轮全通过，%d题三轮全失败，%d题通过与否发生变化；%d题至少通过一次。**'%(report['all_passed_count'],report['all_failed_count'],report['variable_correctness_count'],report['passed_at_least_once_count']),
              '', '## 三轮通过模式','', '`1`为通过，`0`为未通过，位序为第1、2、3轮。','', '| 模式 | 题数 |','|---|---:|']
    for pattern,value in report['patterns'].items():lines.append('| `%s` | %d |'%(pattern,value['count']))
    lines += ['', '## 两两比较','', '| 从→到 | 新增通过 | 失去通过 | 结果不同题数 | 通过集合交集/并集 | Jaccard |','|---|---:|---:|---:|---:|---:|']
    for p in report['pairwise']:
        lines.append('| %d→%d | %d | %d | %d/126 | %d/%d | %.4f |'%(p['from_round'],p['to_round'],len(p['newly_passed_ids']),len(p['lost_pass_ids']),p['correctness_difference_count'],p['pass_intersection_count'],p['pass_union_count'],p['pass_jaccard']))
    lines += ['', '## 哪些题发生变化','', '| 题目 | 模式 | 第1轮 | 第2轮 | 第3轮 |','|---|---|---|---|---|']
    for t in tasks:
        if t['correctness_changed']:
            labels=['通过' if t['rounds'][n]['resolved'] else '失败' for n in ROUNDS]
            labels=['[%s](round-%d/traces/%s.json)'%(label+('（超时）' if t['rounds'][n]['agent_timed_out'] else ''),n,t['instance_id']) for n,label in zip(ROUNDS,labels)]
            lines.append('| `%s` | `%s` | %s | %s | %s |'%(t['instance_id'],t['pattern'],*labels))
    lines += ['', '## 按项目','', '| 项目 | 可评题数 | 第1轮通过 | 第2轮通过 | 第3轮通过 | 结果变化题数 |','|---|---:|---:|---:|---:|---:|']
    for r in report['by_repo']:lines.append('| %s | %d | %d | %d | %d | %d |'%(r['repo'],r['eligible_tasks'],r['resolved_by_round']['1'],r['resolved_by_round']['2'],r['resolved_by_round']['3'],r['variable_correctness_count']))
    lines += ['', '## 对照口径和文件','',
              '- 第1、2轮：GPU3单卡，两路任务；第1轮经历中断恢复，跨两次模型服务。',
              '- 第3轮：GPU2、GPU3两卡，每卡两路，共四路。模型、提示词、预算及seed相同；硬件及调度条件不完全一致。',
              '- 不能将差异单独归因于随机种子、硬件或某项策略；这里报告实际运行的重复一致性。',
              '- `task-results-long.csv`：全部387条任务/轮次记录；排除项明确标记为 `not_scored`。',
              '- `task-comparison.csv`：129题并排指标及原始补丁SHA；`changed-tasks.csv`：其中26题正确性变化项。',
              '- `comparison.json`：全部通过模式的题目ID、两两新增/失去通过的ID、项目统计。',
              '- 补丁SHA比较的是原始生成字节；脱敏公开补丁字节的SHA另由发布清单记录。',
              '- `%d`题三轮补丁原始字节完全一致；补丁不同不代表正确性必然不同。'%report['patches_identical_all_rounds'],
              '- 上述相同补丁中，%d题三轮均为空补丁、%d题为非空补丁；JSON分别列出题目ID。'%(len(report['identical_empty_patch_ids']),len(report['identical_nonempty_patch_ids'])),
              '- 两两比较中未发现“原始补丁相同但通过结果不同”的题；可在 JSON 的 `same_patch_different_correctness_ids` 核对。这不排除不同补丁在语义上等价。',
              '- 预先排除：`fastapi_14077`、`fastapi_14851`（空补丁也通过）；`requests_6757`（环境校准异常）。','']
    return '\n'.join(lines)


def main():
    root=Path(__file__).resolve().parent
    source=json.loads((root/'source-scalars.json').read_text())
    report,tasks=analyze(source)
    if len(source)!=387 or report['eligible_tasks']!=126 or report['total_tasks']!=129:
        raise ValueError('Expected three complete129 runs with126 eligible each')
    out=root/'output';out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    flat=[flatten(t) for t in tasks]
    write_csv(out/'task-comparison.csv',flat,list(flat[0]))
    write_csv(out/'changed-tasks.csv',[r for r in flat if r['correctness_changed']],list(flat[0]))
    long=[]
    for row in sorted(source,key=lambda r:(r['instance_id'],r['round'])):
        long.append({**row,'correctness':'not_scored' if row['status']=='preflight_excluded' else 'passed' if row['resolved'] else 'failed',
                     'verification_exit_codes':'|'.join(str(v) for v in row['verification_exit_codes'])})
    write_csv(out/'task-results-long.csv',long,list(long[0]))
    (out/'COMPARISON.md').write_text(markdown(report,tasks))
    print(json.dumps({k:report[k] for k in ('all_passed_count','all_failed_count','variable_correctness_count','passed_at_least_once_count','patches_identical_all_rounds')},ensure_ascii=False))

if __name__=='__main__':main()
