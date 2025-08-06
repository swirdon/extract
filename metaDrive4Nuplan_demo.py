from metadrive.envs.scenario_env import ScenarioEnv

env = ScenarioEnv(dict(
            use_render=True,    # 是否开启图形界面, 没有图形界面选择 False
            data_directory="/home/amax/nuplan/extrace_dataset_stopline",  # 加载 scenarionet 转换 NuPlan数据的pkl文件夹路径
            start_scenario_index=0,
            num_scenarios=10,
        ))
env.reset()
done = False

while not done:
    obs, reward, done, info, extra = env.step([0.0, 0.0]) # action = [0.0, 0.0] 是 动作无效，replay 模式是根据轨迹走的
    print('finish current step')
    if done:
        env.reset()
        done = False


