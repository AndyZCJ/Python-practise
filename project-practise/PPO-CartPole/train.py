from configs.config import PPOConfig, get_default_config
from trainer.trainer import PPOTrainer, build_trainer
from utils.seed import set_global_seed
from env.cartpole_env import build_cartpole_env
from algorithms.ppo import build_ppo_agent
from models.policy_network import build_policy_network
from models.value_network import build_value_network
from buffers.rollout_buffer import build_rollout_buffer

class TrainingEntry:
    """
    TODO: 组织训练脚本入口。

    你需要完成:
        1. 设置随机种子
        2. 构建训练器及其依赖组件
        3. 启动训练、评估或模型加载流程
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        
        self.env = build_cartpole_env()
        self.policy_network = build_policy_network(self.env.state_dim, self.config.hidden_dim, self.env.action_dim)
        self.value_network = build_value_network(self.env.state_dim, self.config.hidden_dim)
        self.agent = build_ppo_agent(self.config, self.policy_network, self.value_network, self.config.device)
        self.buffer = build_rollout_buffer(self.config)
        self.trainer = build_trainer(self.config, self.env, self.agent, self.buffer)
        # TODO: 如有需要，在这里缓存命令行参数或运行模式

    def run(self) -> PPOTrainer:
        """
        TODO: 串联训练入口逻辑。

        返回:
            trainer: 已构建完成的训练器对象
        """

        # TODO: 根据需要调用 set_global_seed、build_trainer 与 trainer.train
        metrics = self.trainer.train()
        return metrics



def main() -> None:
    """
    TODO: 读取配置并启动训练入口。
    """

    config = get_default_config()
    set_global_seed(config.seed)
    entry = TrainingEntry(config=config)

    # TODO: 根据你的学习计划决定何时调用 entry.run()
    _ = entry.run()


if __name__ == "__main__":
    main()

