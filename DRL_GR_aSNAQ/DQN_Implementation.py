
#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, sys, copy, argparse, random
import matplotlib.pyplot as plt
import math
import os

np.random.seed(10703)
tf.set_random_seed(10703)
random.seed(10703)

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, networkname, trianable):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.
		if environment_name == 'grid':
			self.nObservation = 12
			self.nAction = 6
			self.learning_rate = 0.0001
			self.architecture = [32, 64, 32]

		kernel_init = tf.random_uniform_initializer(-0.5, 0.5)
		bias_init = tf.constant_initializer(0)
		self.input = tf.placeholder(tf.float32, shape=[None, self.nObservation], name='input')
		with tf.variable_scope(networkname):
			layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer1', trainable=trianable)
			layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer2', trainable=trianable)
			layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer3', trainable=trianable)
			self.output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init, bias_initializer=bias_init, name='output', trainable=trianable)

		self.targetQ = tf.placeholder(tf.float32, shape=[None, self.nAction], name='target')
		if trianable == True:
			self.loss = tf.losses.mean_squared_error(self.targetQ, self.output)
			#self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
			#self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
			print("\n\n\n\nAdam\n\n\n\n")
			#################################################################################
			import collections
			self.mL = 8
			self.mF = 100
			self.sk_vec = collections.deque(maxlen=self.mL)
			self.yk_vec = collections.deque(maxlen=self.mL)
			self.memL = collections.deque(maxlen=1)
			self.memF = collections.deque(maxlen=1)
			self.F = collections.deque(maxlen=self.mF)
			self.t_vec = collections.deque(maxlen=1)
			self.iter_k = collections.deque(maxlen=1)
			self.ws_vec = collections.deque(maxlen=1)
			self.vs_vec = collections.deque(maxlen=1)
			self.wo_bar_vec = collections.deque(maxlen=1)
			self.vo_bar_vec = collections.deque(maxlen=1)
			self.old_fun_val = collections.deque(maxlen=1)
			self.mu_val = collections.deque(maxlen=1)
			self.vk_vec = collections.deque(maxlen=1)
			self.alpha_k = collections.deque(maxlen=1)
			self.err = collections.deque(maxlen=1)
			self.iter_k.append(0)
			self.t_vec.append(0)
			self.alpha_k.append(0.01)
			self.mu_val.append(0.1)



			# vo_bar_vec=None, vs_vec=None, alpha_k=1.0,

			self.opt = tf.contrib.opt.ScipyOptimizerInterface(
				self.loss, method='asnaq',
				options={'disp': False, 'sk_vec': self.sk_vec, 'yk_vec': self.yk_vec, 't_vec': self.t_vec,
						 'wo_bar_vec': self.wo_bar_vec, 'vo_bar_vec': self.vo_bar_vec, 'ws_vec': self.ws_vec, 'vs_vec': self.vs_vec,
						 'vk_vec': self.vk_vec, 'F': self.F, 'mu_val': self.mu_val, 'mu_fac': 1.1, 'mu_init': 0.1,'err':self.err,
						 'mu_clip': 0.95, 'alpha_k': self.alpha_k, 'old_fun_val': self.old_fun_val, 'iter': self.iter_k,
						 'memL': self.memL, 'memF': self.memF})

		#################################################################################'''


		with tf.variable_scope(networkname, reuse=True):
			self.w1 = tf.get_variable('layer1/kernel')
			self.b1 = tf.get_variable('layer1/bias')
			self.w2 = tf.get_variable('layer2/kernel')
			self.b2 = tf.get_variable('layer2/bias')
			self.w3 = tf.get_variable('layer3/kernel')
			self.b3 = tf.get_variable('layer3/bias')
			self.w4 = tf.get_variable('output/kernel')
			self.b4 = tf.get_variable('output/bias')


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.memory = []
		self.is_burn_in = False
		self.memory_max = memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		index = np.random.randint(len(self.memory), size=batch_size)
		batch = [self.memory[i] for i in index]

		return batch

	def append(self, transition):
		# Appends transition to the memory.
		self.memory.append(transition)
		if len(self.memory) > self.memory_max:
			self.memory.pop(0)



class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, sess, gridgraph, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.epsilon = 0.05
		
		if environment_name == 'grid':
			self.gamma = 0.9
		self.max_episodes = 500#10000 #20000 #200
		self.batch_size = 32
		self.render = render

		self.qNetwork = QNetwork(environment_name, 'q', trianable=True)
		self.tNetwork = QNetwork(environment_name, 't', trianable=False)
		self.replay = Replay_Memory()

		self.gridgraph = gridgraph

		self.as_w1 = tf.assign(self.tNetwork.w1, self.tNetwork.w1*(1-0.05)+self.qNetwork.w1*0.05)
		self.as_b1 = tf.assign(self.tNetwork.b1, self.tNetwork.b1*(1-0.05)+self.qNetwork.b1*0.05)
		self.as_w2 = tf.assign(self.tNetwork.w2, self.tNetwork.w2*(1-0.05)+self.qNetwork.w2*0.05)
		self.as_b2 = tf.assign(self.tNetwork.b2, self.tNetwork.b2*(1-0.05)+self.qNetwork.b2*0.05)
		self.as_w3 = tf.assign(self.tNetwork.w3, self.tNetwork.w3*(1-0.05)+self.qNetwork.w3*0.05)
		self.as_b3 = tf.assign(self.tNetwork.b3, self.tNetwork.b3*(1-0.05)+self.qNetwork.b3*0.05)
		self.as_w4 = tf.assign(self.tNetwork.w4, self.tNetwork.w4*(1-0.05)+self.qNetwork.w4*0.05)
		self.as_b4 = tf.assign(self.tNetwork.b4, self.tNetwork.b4*(1-0.05)+self.qNetwork.b4*0.05)


		self.init = tf.global_variables_initializer()

		self.sess = sess
		# tf.summary.FileWriter("logs/", self.sess.graph)
		self.sess.run(self.init)
		self.saver = tf.train.Saver(max_to_keep=20)

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		rnd = np.random.rand()
		if rnd <= self.epsilon:
			return np.random.randint(len(q_values))
		else:
			return np.argmax(q_values)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values)


	def network_assign(self):
		# pass the weights of evaluation network to target network
		self.sess.run([self.as_w1, self.as_b1, self.as_w2, self.as_b2, self.as_w3, self.as_b3, self.as_w4, self.as_b4])

	def tempSol(self,solution_combo_filled,gridParameters,sortedHalfWireLength,globali,episode):
		f = open('solutionsDRL/test_benchmark_{dumpBench}_e{epi}.gr.DRLsolution'.format(dumpBench=globali + 1, epi=episode), 'w+')
		# for i in range(1):
		twoPinSolutionPointer = 0
		routeListMerged = solution_combo_filled
		# print('solution_combo_filled',solution_combo_filled)
		for i in range(gridParameters['numNet']):
			singleNetRouteCache = []
			singleNetRouteCacheSmall = []
			indicator = i
			netNum = int(sortedHalfWireLength[i][0])  # i

			i = netNum

			value = '{netName} {netID} {cost}\n'.format(netName=gridParameters['netInfo'][indicator]['netName'],
														netID=gridParameters['netInfo'][indicator]['netID'],
														cost=0)  # max(0,len(routeListMerged[indicator])-1))
			f.write(value)
			for j in range(len(routeListMerged[indicator])):
				for k in range(len(routeListMerged[indicator][j]) - 1):

					a = routeListMerged[indicator][j][k]
					b = routeListMerged[indicator][j][k + 1]

					if (a[3], a[4], a[2], b[3], b[4], b[2]) not in singleNetRouteCache:
						# and (b[3],b[4],b[2]) not in singleNetRouteCacheSmall:
						singleNetRouteCache.append((a[3], a[4], a[2], b[3], b[4], b[2]))
						singleNetRouteCache.append((b[3], b[4], b[2], a[3], a[4], a[2]))
						# singleNetRouteCacheSmall.append((a[3],a[4],a[2]))
						# singleNetRouteCacheSmall.append((b[3],b[4],b[2]))

						diff = [abs(a[2] - b[2]), abs(a[3] - b[3]), abs(a[4] - b[4])]
						if diff[1] > 2 or diff[2] > 2:
							continue
						elif diff[1] == 2 or diff[2] == 2:
							continue
						elif diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
							continue
						elif diff[0] + diff[1] + diff[2] >= 2:
							continue
						else:
							value = '({},{},{})-({},{},{})\n'.format(int(a[0]), int(a[1]), a[2], int(b[0]), int(b[1]),
																	 b[2])
							f.write(value)
				twoPinSolutionPointer = twoPinSolutionPointer + 1
			f.write('!\n')
		f.close()

	def train(self,twoPinNum,twoPinNumEachNet,netSort,savepath,gridParameters,sortedHalfWireLength,globali,model_file=None):
		# ! savepath: "../model_(train/test)"
		# ! if model_file = None, training; if given, testing
    	# ! if testing using training function, comment burn_in in Router.py


		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		# the model will be saved to ../model/
		# the training/testing curve will be saved as a .npz file in ../data/
		if model_file is not None:
			self.saver.restore(self.sess, model_file)

		reward_log = []
		test_reward_log = []
		test_episode = []
		# if not self.replay.is_burn_in:
		# 	self.burn_in_memory()
		solution_combo = []

		reward_plot_combo = []
		reward_plot_combo_pure = []
		training_error = []
		import time
		start_time = time.time()
		end_episode = False
		for episode in np.arange(self.max_episodes*len(self.gridgraph.twopin_combo)):
			#if end_episode:
			#	break
			err_avg = []

			# n_node = len([n.name for n in tf.get_default_graph().as_graph_def().node])
			# print("No of nodes: ", n_node, "\n")

			# print('Route:',self.gridgraph.route)
			solution_combo.append(self.gridgraph.route)

			state, reward_plot, is_best,pinsRouted = self.gridgraph.reset()

			reward_plot_pure = reward_plot-self.gridgraph.posTwoPinNum*100
			# print('reward_plot-self.gridgraph.posTwoPinNum*100',reward_plot-self.gridgraph.posTwoPinNum*100)

			if (episode) % twoPinNum == 0:
				print('\n\n\nTWO PINS ROUTED ',pinsRouted)
				############################################
				if pinsRouted == 114:
					print("\n\n\nTwoPIN", pinsRouted)

					# solution = solution_combo[-twoPinNum:]
					score = self.gridgraph.best_reward
					with open('myResults.txt', 'a') as myFile:
						myFile.write('\n*********Best Reward : {}***************\n'.format(score))

					solution = self.gridgraph.best_route[-twoPinNum:]

					solutionDRL = []

					for i in range(len(netSort)):
						solutionDRL.append([])

					# print('twoPinNum',twoPinNum)
					# print('solution',solution)

					if self.gridgraph.posTwoPinNum == twoPinNum:
						dumpPointer = 0
						for i in range(len(netSort)):
							netToDump = netSort[i]
							for j in range(twoPinNumEachNet[netToDump]):
								# for k in range(len(solution[dumpPointer])):
								solutionDRL[netToDump].append(solution[dumpPointer])
								dumpPointer = dumpPointer + 1
					# print('best reward: ', score)
					# print('solutionDRL: ',solutionDRL,'\n')
					else:
						solutionDRL = solution

					## Generte solution

					# print ('solution_combo: ',solution_combo)
					self.tempSol(solutionDRL, gridParameters, sortedHalfWireLength, globali, episode/20)

				############################################
				reward_plot_combo.append(reward_plot)
				reward_plot_combo_pure.append(reward_plot_pure)
				#if (time.time() - start_time) > 3800:
					#end_episode = True
					#break
			is_terminal = False
			rewardi = 0.0
			if episode % 10 == 0:
				self.network_assign()

			rewardfortwopin = 0
			while not is_terminal:
				observation = self.gridgraph.state2obsv()
				q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
				action = self.epsilon_greedy_policy(q_values)
				# print(action)
				nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
				# print(nextstate)
				observation_next = self.gridgraph.state2obsv()
				self.replay.append([observation, action, reward, observation_next, is_terminal])
				state = nextstate
				rewardi = rewardi+reward
				rewardfortwopin = rewardfortwopin + reward

				batch = self.replay.sample_batch(self.batch_size)
				batch_observation = np.squeeze(np.array([trans[0] for trans in batch]))
				batch_action = np.array([trans[1] for trans in batch])
				batch_reward = np.array([trans[2] for trans in batch])
				batch_observation_next = np.squeeze(np.array([trans[3] for trans in batch]))
				batch_is_terminal = np.array([trans[4] for trans in batch])
				q_batch = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: batch_observation})
				q_batch_next = self.sess.run(self.tNetwork.output, feed_dict={self.tNetwork.input: batch_observation_next})
				#y_batch = batch_reward+self.gamma*(1-batch_is_terminal)*np.max(q_batch_next, axis=1)

				#Double DQN
				#q_batch_next_prime = self.sess.run(self.qNetwork.output,feed_dict={self.qNetwork.input: batch_observation_next})
				#q_batch_prime = self.sess.run(self.tNetwork.output,feed_dict={self.tNetwork.input: batch_observation})
				#next_action = np.argmax(q_batch_prime,1)
				#y_batch = batch_reward+self.gamma*(1-batch_is_terminal)*q_batch_next_prime[np.arange(self.batch_size),next_action]

				#q_batch_prime = self.sess.run(self.qNetwork.output,feed_dict={self.tNetwork.input: batch_observation})
				q_batch_next_prime = self.sess.run(self.qNetwork.output,feed_dict={self.qNetwork.input: batch_observation_next})
				next_action = np.argmax(q_batch_next_prime, 1)
				y_batch = batch_reward+self.gamma*(1-batch_is_terminal)*q_batch_next[np.arange(self.batch_size),next_action]

				targetQ = q_batch.copy()
				targetQ[np.arange(self.batch_size), batch_action] = y_batch
				#_, train_error = self.sess.run([self.qNetwork.opt, self.qNetwork.loss], feed_dict={self.qNetwork.input: batch_observation, self.qNetwork.targetQ: targetQ})


				_ = self.qNetwork.opt.minimize(self.sess, feed_dict={self.qNetwork.input: batch_observation, self.qNetwork.targetQ: targetQ})
				train_error = self.qNetwork.err[0]
				self.gridgraph.training_error.append(train_error)
			reward_log.append(rewardi) #comment in test; do not save model test

			self.gridgraph.instantrewardcombo.append(rewardfortwopin)

			# print(episode, rewardi)
			'''
			if is_best == 1:
				#print('self.gridgraph.route',self.gridgraph.route)
				print('Save model')
				test_reward = self.test(no=10)
				print(test_reward)
				test_reward_log.append(test_reward/10.0)
				test_episode.append(episode)
				save_path = self.saver.save(self.sess, "{}/model_{}.ckpt".format(savepath,episode))
				print("Model saved in path: %s" % savepath)'''
			### Change made
			# if rewardi >= 0:
				# print(self.gridgraph.route)
				# solution_combo.append(self.gridgraph.route)

		# solution = solution_combo[-twoPinNum:]
		score = self.gridgraph.best_reward
		with open('myResults.txt', 'a') as myFile:
			myFile.write('\n*********Best Reward : {}***************\n'.format(score))

		solution = self.gridgraph.best_route[-twoPinNum:]

		solutionDRL = []

		for i in range(len(netSort)):
			solutionDRL.append([])

		#print('twoPinNum',twoPinNum)
		#print('solution',solution)

		if self.gridgraph.posTwoPinNum  == twoPinNum:
			dumpPointer = 0
			for i in range(len(netSort)):
				netToDump = netSort[i]
				for j in range(twoPinNumEachNet[netToDump]):
					# for k in range(len(solution[dumpPointer])):
					solutionDRL[netToDump].append(solution[dumpPointer])
					dumpPointer = dumpPointer + 1
		# print('best reward: ', score)
		# print('solutionDRL: ',solutionDRL,'\n')
		else:
			solutionDRL = solution

		## Generte solution

		# print ('solution_combo: ',solution_combo)


		#
		print(test_reward_log)
		train_episode = np.arange(self.max_episodes)
		np.savez('../data/training_log.npz', test_episode=test_episode, test_reward_log=test_reward_log,reward_log=reward_log, train_episode=train_episode)

		self.sess.close()
		tf.reset_default_graph()

		return solutionDRL,reward_plot_combo,reward_plot_combo_pure,solution,self.gridgraph.posTwoPinNum



	def test(self, model_file=None, no=20, stat=False):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		# uncomment this line below for videos
		# self.env = gym.wrappers.Monitor(self.env, "recordings", video_callable=lambda episode_id: True)
		if model_file is not None:
			self.saver.restore(self.sess, model_file)
		reward_list = []
		cum_reward = 0.0
		for episode in np.arange(no):
			episode_reward = 0.0
			state = self.gridgraph.reset()
			is_terminal = False
			while not is_terminal:
				observation = self.gridgraph.state2obsv()
				q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
				action = self.greedy_policy(q_values)
				nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
				state = nextstate
				episode_reward = episode_reward+reward
				cum_reward = cum_reward+reward
			reward_list.append(episode_reward)
		if stat:
			return cum_reward, reward_list
		else:
			return cum_reward

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		print('Start burn in...')
		state = self.gridgraph.reset()
		for i in np.arange(self.replay.burn_in):
			if i % 2000 == 0:
				print('burn in {} samples'.format(i))
			observation = self.gridgraph.state2obsv()
			action = self.gridgraph.sample()
			nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
			observation_next = self.gridgraph.state2obsv()
			self.replay.append([observation, action, reward, observation_next, is_terminal])
			if is_terminal:
				# print(self.gridgraph.current_step)
				state = self.gridgraph.reset()
			else:
				state = nextstate
		self.replay.is_burn_in = True
		print('Burn in finished.')

	def burn_in_memory_search(self,observationCombo,actionCombo,rewardCombo,
        observation_nextCombo,is_terminalCombo): # Burn-in with search
		print('Start burn in with search algorithm...')
		for i in range(len(observationCombo)):
			observation = observationCombo[i]
			action = actionCombo[i]
			reward = rewardCombo[i]
			observation_next = observation_nextCombo[i]
			is_terminal = is_terminalCombo[i]

			self.replay.append([observation, action, reward, observation_next, is_terminal])

		self.replay.is_burn_in = True
		print('Burn in with search algorithm finished.')

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--test',dest='test',type=int,default=0)
	parser.add_argument('--lookahead',dest='lookahead',type=int,default=0)
	parser.add_argument('--test_final',dest='test_final',type=int,default=0)
	parser.add_argument('--model_no',dest='model_file_no',type=str)
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	#gpu_ops = tf.GPUOptions(allow_growth=True)
	#config = tf.ConfigProto(gpu_options=gpu_ops)
	#sess = tf.Session(config=config)
	sess = tf.Session()

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
	model_path = '../model/'
	data_path = '../data/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	agent = DQN_Agent(environment_name, sess, render=args.render)
	if args.train == 1:
		agent.train()
	if args.test == 1:
		print(agent.test(model_file="../model/model_{}.ckpt".format(args.model_file_no))/20.0)
	sess.close()


if __name__ == '__main__':
	main(sys.argv)

