"""Bound the audited calibrated runner to per-task effective budget prompts."""
import gemmacodeagent_mm_task as legacy
from pathlib import Path
import shutil

_ORIGINAL = legacy.adapt_source

def materialize_native_submission(source,destination,raw):
 from budget_contract import materialize_submission
 output,metadata=materialize_submission(source,destination,raw)
 for path in list(Path(output).iterdir()):
  if path.is_symlink():
   expected=(Path(source)/path.name).resolve()
   if path.resolve()!=expected:raise ValueError('Unexpected submission link target')
   staging=Path(output)/('.native-'+path.name+'.tmp')
   if expected.is_dir():shutil.copytree(expected,staging,symlinks=False)
   else:shutil.copy2(expected,staging)
   path.unlink();staging.replace(path)
 for path in Path(output).rglob('*'):
  if path.is_symlink() or not path.resolve().is_relative_to(Path(output).resolve()):
   raise ValueError('Effective submission must be self-contained')
 return output,metadata

def assert_agent_execution(row):
 if str(row.get('error_message') or '').startswith('Sandbox execution error:'):
  raise RuntimeError('Agent sandbox setup failed; this is not a model score')

def replace_once(source,before,after):
 if source.count(before)!=1:raise ValueError('Audited effective-budget anchor changed')
 return source.replace(before,after,1)

def adapt_source(source,context):
 adapted=_ORIGINAL(source,context)
 before='    raw=yaml.safe_load((submission/"eval_config.yaml").read_text()).get("evaluation",{})'
 after=before+'\n    from gemmacodeagent_morning25_task import materialize_native_submission\n    submission,budget_contract=materialize_native_submission(submission,out/"effective_submission",raw)\n    from adk_submission.discovery import validate_directory\n    validate_directory(submission)\n    effective_text=(submission/"prompts/system.md").read_text()\n    if effective_text.count("## Budget: 56 tool calls, 15 minutes")!=1 or effective_text.count("You have at most 56 tool calls and 15 minutes for this task.")!=1: raise ValueError("Effective prompt budget mismatch")'
 adapted=replace_once(adapted,before,after)
 adapted=replace_once(adapted,'manifest={"mode":args.mode,','manifest={"budget_contract":budget_contract,"mode":args.mode,')
 before="extra_body={'thinking_token_budget': 6000, 'chat_template_kwargs': {'enable_thinking': True}}"
 after="extra_body={'thinking_token_budget': 6000, 'top_k': 64, 'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': False}}"
 if adapted.count(before)!=2:raise ValueError('Audited request route count changed')
 adapted=adapted.replace(before,after)
 anchor='                row=result.model_dump(mode="json",exclude={"trace"})'
 return replace_once(adapted,anchor,anchor+'\n                from gemmacodeagent_morning25_task import assert_agent_execution\n                if args.mode=="agent": assert_agent_execution(row)')

def main():
 legacy.adapt_source=adapt_source
 legacy.main()

if __name__=='__main__':main()
