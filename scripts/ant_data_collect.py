import tensorflow as tf

from airl.algos.trpo import TRPO
from airl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from airl.envs.env_utils import CustomGymEnv
from airl.utils.log_utils import rllab_logdir
from airl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial


def main(exp_name, ent_wt=1.0):
    tf.reset_default_graph()
    env = TfEnv(CustomGymEnv('airl/CustomAnt-v0', record_video=False, record_log=False))
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    with tf.Session(config=get_session_config()) as sess:
        algo = TRPO(
            env=env,
            policy=policy,
            n_itr=1500,
            batch_size=20000,
            max_path_length=500,
            discount=0.99,
            store_paths=True,
            entropy_weight=ent_wt,
            baseline=LinearFeatureBaseline(env_spec=env.spec)
        )
        with rllab_logdir(algo=algo, dirname='data/ant_data_collect/%s'%exp_name):
            algo.train(sess=sess)

if __name__ == "__main__":
    params_dict = {
        'ent_wt': 0.1,
        'exp_name': 'test1'
    }
    # run_sweep_parallel(main, params_dict, repeat=4)
    main(**params_dict)
