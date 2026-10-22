"""以当前环境的空间定义加载策略，避免跨 NumPy 版本反序列化空间。"""
def load_policy(path, env):
    """权重不变；用已校验环境空间替换训练机保存的空间对象。"""
    from stable_baselines3 import PPO
    return PPO.load(path, device='cpu', custom_objects={
        'observation_space': env.observation_space,
        'action_space': env.action_space,
    })
