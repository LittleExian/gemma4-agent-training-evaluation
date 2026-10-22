"""Render the two budget statements from the effective evaluator configuration."""
import hashlib,json,math,re,shutil
from pathlib import Path

VERSION='evaluation-budget-v2'
HEADER=re.compile(r'^## Budget: [0-9.]+ tool calls, [0-9.]+ minutes$',re.M)
LEGACY=re.compile(r'^- [0-9.]+ tool calls and [0-9.]+ minutes in total\.',re.M)
SENTENCE=re.compile(r'You have at most [0-9.]+ tool calls and [0-9.]+ minutes for this task\.')

def render_budget(text,raw):
    calls=raw.get('max_tool_calls',28);minutes=raw.get('max_time_minutes',8)
    if isinstance(calls,bool) or not isinstance(calls,int) or calls<=0:
        raise ValueError('max_tool_calls must be a positive integer')
    if isinstance(minutes,bool) or not isinstance(minutes,(int,float)) or not math.isfinite(minutes) or minutes<=0:
        raise ValueError('max_time_minutes must be positive and finite')
    if len(LEGACY.findall(text))==1 and not HEADER.search(text) and not SENTENCE.search(text):
        return LEGACY.sub(f'- {calls} tool calls and {minutes:g} minutes in total.',text)
    if len(HEADER.findall(text))!=1 or len(SENTENCE.findall(text))!=1:
        raise ValueError('Unknown system budget format: refusing a silent mismatch')
    text=HEADER.sub(f'## Budget: {calls} tool calls, {minutes:g} minutes',text)
    return SENTENCE.sub(f'You have at most {calls} tool calls and {minutes:g} minutes for this task.',text)

def materialize_submission(source,destination,raw):
    """Each task gets its own effective prompt; shared submissions stay immutable.

    Other config/adapter paths are linked, so this does not duplicate LoRA weights.
    An old run with a different effective prompt cannot be silently resumed.
    """
    source=Path(source).resolve();destination=Path(destination)
    text=(source/'prompts/system.md').read_text()
    fixed=render_budget(text,raw)
    if destination.exists():
        if (destination/'prompts/system.md').read_text()!=fixed:
            raise ValueError('Effective system prompt changed; use a NEW run directory')
    else:
        destination.mkdir(parents=True)
        for child in source.iterdir():
            if child.name=='prompts':
                for item in [child,*child.rglob('*')]:
                    dest=destination/item.relative_to(source)
                    if item.is_dir():dest.mkdir(parents=True,exist_ok=True)
                    else:shutil.copyfile(item,dest)
            else:(destination/child.name).symlink_to(child,target_is_directory=child.is_dir())
        (destination/'prompts/system.md').write_text(fixed)
    return destination,{'version':VERSION,'max_tool_calls':raw.get('max_tool_calls',28),
        'max_time_minutes':raw.get('max_time_minutes',8),
        'source_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'effective_prompt_sha256':hashlib.sha256(fixed.encode()).hexdigest()}
