import os
import time
os.environ["CLEARML_WEB_HOST"] = "https://prod-clearml-webserver.srv.deeproute.cn"
os.environ["CLEARML_API_HOST"] = "https://prod-clearml-apiserver.srv.deeproute.cn"
os.environ["CLEARML_FILES_HOST"] = "https://prod-clearml-fileserver.srv.deeproute.cn"

# enable deekeeper for experiment management
enable_deekeeper = False
if "CLEARML_API_ACCESS_KEY" and "CLEARML_API_SECRET_KEY" in os.environ:
    try:
        from clearml import Task, Alarm, TaskAlarmTypesEnum, Logger
        enable_deekeeper = True
    except ImportError:
        print('"pip install dr-clearml" first!')
else:
    print("""Skip deekeeper... Access AK and SK for deekeeper is not generated!
          Please generate them from HERE(https://deekeeper-fe.srv.deeproute.cn/settings/workspace)""")

# deekeeper
def refine_exp_name(task_name):
    # to avoid the same experiment name
    return task_name + time.strftime('_%Y%m%d-%H%M%S',time.localtime(int(time.time())))

def get_alarm_type(is_alarm=True):
    if is_alarm:
        return Alarm(alarm_type=TaskAlarmTypesEnum.all_alarm)
    else:
        return None
 
def init_deekeeper_task(project_name, task_name, feishu_alarm=True):
    
    if not enable_deekeeper:
        print("dr-clearml not install or check the environ of CLEARML")
        return None
    
    clearml_task = Task.init(project_name=project_name, 
                        task_name=refine_exp_name(task_name),
                        alarm=get_alarm_type(feishu_alarm),
                        auto_connect_frameworks=dict(pytorch=["*.pth"])
                        )
    
    return clearml_task

def get_deekeeper_logger():
    if not enable_deekeeper:
        print("dr-clearml not install or check the environ of CLEARML")
        return None
    return Logger.current_logger()