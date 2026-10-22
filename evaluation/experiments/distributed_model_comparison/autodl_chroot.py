
import os,subprocess,uuid,shutil
from pathlib import Path
from swegemma.sandbox.subprocess import SubprocessManager
from adk_eval_core.sandbox import execute_subprocess_command
P=Path(os.environ['CODEAGENT_ROOT'])
class ChrootManager(SubprocessManager):
 def __init__(self,timeout_seconds=300,**kw):
  super().__init__(timeout_seconds=timeout_seconds,base_dir=P/'tmp/chroot_tasks',system_site_packages=False)
 @property
 def work_dir(self):return '/workspace'
 def start(self):
  sid=str(uuid.uuid4())[:12];root=self.base_dir/sid;root.mkdir(parents=True)
  subprocess.run(['cp','-a','--reflink=auto',str(P/'sandbox/rootfs')+'/.',str(root)],check=True)
  for x in ['workspace','tmp','wheels','dev','proc']: (root/x).mkdir(exist_ok=True)
  null=root/'dev/null'
  if not null.exists():
   os.mknod(null,0o666|0o20000,os.makedev(1,3))
  with self._lock:
   self._sandboxes[sid]={'root':root,'workspace':root/'workspace','tmp':root/'tmp','wheels':root/'wheels','venv':root/'usr/local'}
   if self._default_sandbox_id is None:self._default_sandbox_id=sid
  return sid
 def exec(self,sandbox_id,command,*,timeout=None):
  root=self._sandboxes[sandbox_id]['root']
  env={'PATH':'/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','HOME':'/root','LANG':'C.UTF-8','TMPDIR':'/tmp','PIP_NO_INDEX':'1','PIP_FIND_LINKS':'/wheels','PYTHONUNBUFFERED':'1'}
  args=['/usr/sbin/chroot',str(root),'/usr/bin/setpriv','--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','--no-new-privs','/bin/bash','-c','cd /workspace && '+command]
  return execute_subprocess_command(command=args,cwd=root,env=env,timeout=timeout or self.timeout_seconds,shell=False)

 def extract_archive_to_container(self,sandbox_id,archive,destination):
  import shlex
  staged='/tmp/import_'+uuid.uuid4().hex+'.tar'
  self.copy_to(sandbox_id,Path(archive),staged)
  command='mkdir -p '+shlex.quote(destination)+' && tar --no-same-owner -xf '+shlex.quote(staged)+' -C '+shlex.quote(destination)+' && rm -f '+shlex.quote(staged)
  return self.exec(sandbox_id,command).exit_code==0
