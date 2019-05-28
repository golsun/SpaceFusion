from shared import *
from tf_lib import *
from dataset import *

"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""


class LossHistory(Callback):
	def reset(self):
		self.losses = []

	def on_train_begin(self, logs={}):
		self.reset()

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))



def extract_layers(model):
	layers = dict()
	for layer in model.layers:
		layers[layer.name] = layer
	return layers



def write_log(path, s, PRINT=True, mode='a'):
	if PRINT:
		print(s)
	if not s.endswith('\n'):
		s += '\n'
	while True:
		try:
			with open(path, mode) as f:
				f.write(s)
			break
		except:# PermissionError as e:
			#print(e)
			print('sleeping...')
			time.sleep(2)




class ModelWrapper:

	def init(self):

		self.history = LossHistory()
		self.tsv_train = os.path.join(self.fld, self.model_name+'_train.tsv')
		self.log_train = os.path.join(self.fld, self.model_name+'_train.log')
		self.log_eval = os.path.join(self.fld, self.model_name+'_eval.log')
		self.word2vec = None
		self.vali_data = None

		if self.new and os.path.exists(self.fld):
			print('%s\nalready exists, do you want to delete the folder? (y/n)'%self.fld)
			ans = input()
			if ans.lower() == 'y':
				shutil.rmtree(self.fld)
				time.sleep(2)	# otherwise the following makedirs may fail
				print('fld deleted')

		makedirs(os.path.join(self.fld, 'epochs'))		
		self.build_model()
		self.build_model_test()

		if self.new:
			open(self.tsv_train, 'w')
			open(self.log_train, 'w')
			with open(os.path.join(self.fld, self.model_name+'.arch.yaml'), 'w') as f:
				f.write(self.model.to_yaml())


	def plot_model(self, model=None, fname=''):

		if SKIP_VIS:
			return

		if bool(fname):
			fname = '_' + fname
		fname = self.model_name + fname
		if model is None:
			model = self.model
		while True:
			try:
				plot_model(model, 
					to_file=os.path.join(self.fld, fname + '.png'), 
					show_shapes=True)
				break
			except PermissionError as e:
				print(e)
				time.sleep(2)


	def train(self, 
		batch_size, epochs, 
		batch_per_load=100,
		skip=0,
		epoch_init=0,
		):

		write_log(self.tsv_train, self.tsv_header, PRINT=False)
		self.n_trained = skip

		for epoch in range(epoch_init, epochs):
			self.dataset.reset('train')
			if epoch == epoch_init: 
				self.dataset.skip(skip)
			while True:
				s = '\n***** Epoch %i/%i, trained %.2fM *****'%(
					epoch + 1, epochs, self.n_trained/1e6)
				write_log(self.log_train, s)
				n = self.train_a_load(batch_size, batch_per_load)
				if n == 0:
					break
				self.save_weights('%s_epoch%i'%(self.model_name, epoch + 1), subfld=True)
		self.save_weights(self.model_name)


	def _compile(self, model=None, loss=None, loss_weights=None):
		if model is None:
			model = self.model
		if loss is None:
			loss = self.loss
		model.compile(optimizer=Adam(lr=self.lr), loss=loss, loss_weights=loss_weights)


	def load_weights(self, path=''):
		if path == '':
			path = os.path.join(self.fld, self.model_name + '.weight.h5')
		print('loading weights from %s'%path)
		self.model.load_weights(path)


	def build_model_test(self):
		pass


	def save_weights(self, fname, subfld=False):
		fname += '.weight.h5'
		if subfld:
			path = os.path.join(self.fld, 'epochs', fname)
		else:
			path = os.path.join(self.fld, fname)
		path_old = path + '.old'
		n_try = 0
		max_try = 5
		while n_try < max_try:
			try:
				n_try += 1
				if os.path.exists(path_old):
					print('deleting ' + path_old)
					os.remove(path_old)
					time.sleep(1)
				if os.path.exists(path):
					print('moving to ' + path_old)
					shutil.move(path, path_old)
					time.sleep(1)
				print('saving...')
				self.model.save_weights(path)
				print('saved to: '+path)
				return
			except Exception as e:
				print('vvvvvvvvvvv handled exception vvvvvvvvvvv')
				print(e)
				print('^^^^^^^^^^^ handled exception ^^^^^^^^^^^')
				time.sleep(2)
		print('could not save weights')


	def _get_vali_data(self):
		self.dataset.reset('vali')
		self.vali_data = self.dataset.load_data('vali', max_n=2000, check_src=True)
		self.dataset.reset('vali')




class Seq2SeqBase(ModelWrapper):

	def _stacked_rnn(self, rnns, inputs, initial_states=None):
		if initial_states is None:
			initial_states = [None] * len(rnns)

		outputs, state = rnns[0](inputs, initial_state=initial_states[0])
		states = [state]
		for i in range(1, len(rnns)):
			outputs, state = rnns[i](outputs, initial_state=initial_states[i])
			states.append(state)
		return outputs, states


	def _build_encoder(self, inputs, layers, prefix=None):
		if prefix is None:
			prefix = self.prefix[0]
		encoder_outputs, encoder_states = self._stacked_rnn(
				[layers['%s_encoder_rnn_%i'%(prefix, i)] for i in range(self.encoder_depth)], 
				layers['embedding'](inputs))
		latent = encoder_states[-1]
		return latent


	def _build_decoder(self, input_seqs, input_states, layers):
		"""
		for auto-regressive, states are returned and used as input for the generation of the next token
		for teacher-forcing, token already given, so only need init states
		"""

		decoder_outputs, decoder_states = self._stacked_rnn(
				[layers['decoder_rnn_%i'%i] for i in range(self.decoder_depth)], 
				layers['embedding'](input_seqs),
				input_states)
		decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
		decoder_outputs = layers['decoder_softmax'](decoder_outputs)
		return decoder_outputs, decoder_states


	def _create_layers(self):
		layers = dict()

		if self.new:
			self._load_word2vec()
		if self.word2vec is None:
			emb_init = 'uniform'
		else:
			emb_init = keras.initializers.Constant(value=self.word2vec)

		layers['embedding'] = Embedding(
				self.dataset.num_tokens + 1,		# +1 as mask_zero 
				self.token_embed_dim, mask_zero=True, 
				embeddings_initializer=emb_init,
				name='embedding')

		for i in range(self.decoder_depth):
			name = 'decoder_rnn_%i'%i
			layers[name] = GRU(
				self.rnn_units, 
				return_state=True,
				return_sequences=True, 
				name=name)

		for prefix in self.prefix:
			for i in range(self.encoder_depth):
				name = '%s_encoder_rnn_%i'%(prefix, i)
				layers[name] = GRU(
						self.rnn_units, 
						return_state=True,
						return_sequences=True, 
						name=name)

		layers['decoder_softmax'] = Dense(
			self.dataset.num_tokens + 1, 		# +1 as mask_zero
			activation='softmax', name='decoder_softmax')

		return layers


	def _build_model_test(self):

		# layers 
		layers = extract_layers(self.model)
		decoder_inputs = Input(shape=(None,), name='decoder_inputs')
		
		# connections: encoder

		self.model_tf = dict()
		self.model_encoder = dict()
		latent_prefix = dict()
		for prefix in self.prefix:
			encoder_inputs = Input(shape=(None,), name=prefix+'_encoder_inputs')
			latent = self._build_encoder(encoder_inputs, layers, prefix=prefix)
			self.model_encoder[prefix] = Model(encoder_inputs, latent)
			#self.plot_model(self.model_encoder[prefix], prefix+'_encoder')
			
			if self.model_name != 'mtask':
				continue

		# connections: autoregressive

		decoder_inital_states = []
		for i in range(self.decoder_depth):
			decoder_inital_states.append(Input(shape=(self.rnn_units,), name="decoder_inital_state_%i"%i))
		decoder_outputs, decoder_states = self._build_decoder(decoder_inputs, decoder_inital_states, layers)
		model_decoder_ar = Model([decoder_inputs] + decoder_inital_states, [decoder_outputs] + decoder_states)
		self.decoder_ar = DecoderAR(self.dataset, model_decoder_ar, self.decoder_depth)

		#self.plot_model(self.decoder_ar.model, 'decoder_AR')


		# connections: teacher-forcing

		latent =  Input(shape=(self.rnn_units,), name='latent')
		decoder_outputs, _ = self._build_decoder(decoder_inputs, [latent]*self.decoder_depth, layers)
		self.model_decoder_tf = Model([latent, decoder_inputs], decoder_outputs)
		self._compile(self.model_decoder_tf, loss=DECODER_LOSS)


	
	def _load_word2vec(self):
		if self.path_word2vec is None:
			return

		print('loading word2vec from '+self.path_word2vec)
		n_words = len(self.dataset.token2index) + 1		# +1 for PAD
		self.word2vec = np.random.random((n_words, self.token_embed_dim)) * 0.1

		n_covered = 0
		for line in open(self.path_word2vec, encoding='utf-8'):
			w, vec = line.split(" ", 1)
			if w in self.dataset.token2index:
				n_covered += 1
				self.word2vec[self.dataset.token2index[w], :] = np.fromstring(vec, sep=" ")

		not_covered = n_words - n_covered
		print('%i (%.2f perc) not covered'%(not_covered, 100*not_covered/n_words))

		

	def dialog(self, input_text, prefix=None, method='greedy', beam_width=10):
		if prefix is None:
			prefix = self.prefix[0]
		source_seq = self.dataset.txt2seq(input_text)
		latent = self.model_encoder[prefix].predict(np.atleast_2d(source_seq))
		if method=='greedy':
			return self.decoder_ar.predict(latent)
		elif method=='sampling':
			return self.decoder_ar.predict(latent, sampling=True)
		elif method=='beam':
			return self.decoder_ar.predict_beam(latent, beam_width=beam_width)
		elif method=='rand':
			latent = rand_latent(latent, 1.5, limit=True)
			return self.decoder_ar.predict(latent)
		else:
			raise ValueError





class Seq2Seq(Seq2SeqBase):

	"""
	simple seq2seq and a optional 1-to-1 thinker (MLP)
	"""

	def __init__(self, 
		dataset, 
		fld, 
		token_embed_dim, 
		rnn_units, 
		encoder_depth, 
		decoder_depth,
		dropout_rate=0.,
		prefix='dial',
		lr=1e-4,
		new=False,
		path_word2vec=None):

		self.fld = fld
		self.token_embed_dim = token_embed_dim
		self.rnn_units = rnn_units
		self.encoder_depth = encoder_depth
		self.decoder_depth = decoder_depth
		self.dropout_rate = dropout_rate
		self.dataset = dataset
		self.model_name = prefix
		self.lr = lr
		self.prefix = [prefix]
		self.path_word2vec = path_word2vec

		self.tsv_header = 'lr\tn_trained\tt0\tdt\tloss_train\tloss_vali'
		self.loss = DECODER_LOSS
		self.loss_weights = None
		print('new = %s'%new)
		self.new = new
		self.init()


			
	def build_model(self):

		# layers

		encoder_inputs = Input(shape=(None,), name='encoder_inputs')
		decoder_inputs = Input(shape=(None,), name='decoder_inputs')
		layers = self._create_layers()	

		# connections: teacher forcing

		latent = self._build_encoder(encoder_inputs, layers)
		decoder_outputs, _ = self._build_decoder(decoder_inputs, [latent]*self.decoder_depth, layers)

		# models

		self.model = Model(
				[encoder_inputs, decoder_inputs], 	# [input sentences, ground-truth target sentences],
				decoder_outputs)					# shifted ground-truth sentences 

		self._compile()
		self.plot_model()


	def build_model_test(self):
		self._build_model_test()


	def train_a_load(self, batch_size, batch_per_load):
		encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts = \
					self.dataset.load_data('train', batch_size * batch_per_load)

		n_sample = len(target_texts)
		if n_sample == 0:
			return 0

		t0 = datetime.datetime.now()
		t0_str = str(t0).split('.')[0]
		s = '\nstart:\t%s'%t0_str
		write_log(self.log_train, s)

		self.model.fit(
			[encoder_input_data[self.prefix[0]], decoder_input_data], 
			decoder_target_data,
			batch_size=batch_size,
			callbacks=[self.history],
			verbose=FIT_VERBOSE)
		self.n_trained += n_sample

		dt = (datetime.datetime.now() - t0).seconds
		loss = np.mean(self.history.losses)

		# vali --------------------

		if self.vali_data is None:
			self._get_vali_data()
		encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts = self.vali_data
		loss_vali = self.model.evaluate(
			[encoder_input_data[self.prefix[0]], decoder_input_data], 
			decoder_target_data,
			verbose=0)

		s = 'spent:\t%s sec\ntrain:\t%.4f\nvali:\t%.4f'%(dt, loss, loss_vali)
		write_log(self.log_train, s)
		write_log(self.tsv_train, '\t'.join(map(strmap, 
			['%.1e'%self.lr, '%.3f'%(self.n_trained/1e6), t0_str, dt, '%.4f'%loss, '%.4f'%loss_vali])), PRINT=False)

		if SKIP_VIS:
			return n_sample

		self.build_model_test()
		for i in range(5):
			if self.prefix[0] == 'dial':
				src = source_texts[i]
			else:
				src = target_texts[i]
			NNL, predicted = self.dialog(src)
			s = '\n'.join([
				'-'*10,
				'source: \t'+src,
				'target: \t'+target_texts[i],
				'pred:   \t'+predicted,
				])
			write_log(self.log_train, s)
		
		return n_sample



# -------------------------------------------------------------

def _batch_spread(_, y_pred):
	# capped difference between each pair of y_pred

	CAP = 0.3
	expanded_a = tf.expand_dims(y_pred, 1)
	expanded_b = tf.expand_dims(y_pred, 0)
	d_squared = tf.reduce_mean(tf.squared_difference(expanded_a, expanded_b), 2)
	d = tf.sqrt(tf.maximum(0., d_squared))
	return tf.reduce_mean(tf.minimum(CAP, d))
	#score = tf.minimum(0.3**2, d_squared)
	#return tf.sqrt(tf.reduce_mean(score))


def _sqrt_mse(_, y_pred):
	return tf.sqrt(tf.reduce_mean(tf.pow(y_pred, 2)))


def _add_noise(mu, stddev):
	eps = K.random_normal(shape=K.shape(mu))
	return mu + tf.multiply(eps, stddev)

def _interp(ab):
	a, b = ab
	u = K.random_uniform(shape=(K.shape(a)[0], 1))
	u = K.tile(K.reshape(u, [-1,1]), [1, K.shape(a)[1]])	# repeat along axis=1
	return a + tf.multiply(b - a, u)


class MTask(Seq2SeqBase):
	# the SpaceFusion model, multi-task with regularizers

	def __init__(self, 
		dataset, 
		fld, 
		token_embed_dim, 
		rnn_units, 
		encoder_depth, 
		decoder_depth,
		dropout_rate=0.,
		stddev=0.1,
		wt_S2T=0.,
		wt_S2S=0.,
		wt_T2T=0.,
		interp=True,
		lr=1e-4,
		new=False,
		path_word2vec=None,
		):

		self.fld = fld
		self.token_embed_dim = token_embed_dim
		self.rnn_units = rnn_units
		self.encoder_depth = encoder_depth
		self.decoder_depth = decoder_depth
		self.dropout_rate = dropout_rate
		self.dataset = dataset
		self.stddev = stddev
		self.interp = interp
		self.lr = lr
		self.path_word2vec = path_word2vec

		self.loss = [_sqrt_mse, _batch_spread, _batch_spread]
		self.loss_weights = [wt_S2T, - wt_S2S, - wt_T2T]	# shrink d(S,T) but encourage d(S,S) and d(T,T)
		n = 2 + self.interp
		self.loss += [DECODER_LOSS] * n
		self.loss_weights += [1./n] * n

		print('loss: '+'-'*20)
		for i in range(len(self.loss)):
			print('  %.2f %s'%(self.loss_weights[i], self.loss[i]))
		print('-'*20)

		self.model_name = 'mtask'
		self.prefix = ['auto','dial']
		self.tsv_header = 'lr\tn_trained\tt0\tdt\tloss_train\tloss_vali\td(T,S)\td(S,S)\td(T,T)\tloss_auto\tloss_dial\tloss_interp'
		self.new = new
		self.init()


	def build_model(self):

		# layers

		auto_encoder_inputs = Input(shape=(None,), name='auto_encoder_inputs')
		dial_encoder_inputs = Input(shape=(None,), name='dial_encoder_inputs')
		decoder_inputs = Input(shape=(None,), name='decoder_inputs')
		layers = self._create_layers()	# create new
		inputs = [auto_encoder_inputs, dial_encoder_inputs, decoder_inputs]

		# connections: teacher forcing

		vec_tgt = self._build_encoder(auto_encoder_inputs, layers, prefix='auto')
		vec_src = self._build_encoder(dial_encoder_inputs, layers, prefix='dial')
		diff_S2T = Subtract(name='diff_S2T')([vec_src, vec_tgt])		# [batch, rnn_units]
		outputs = [diff_S2T, vec_src, vec_tgt]
		
		if self.stddev > 0:
			noisy = Lambda(_add_noise, 
				arguments={'stddev':self.stddev}, 
				name='noisy')
			vec_src = noisy(vec_src)
			vec_tgt = noisy(vec_tgt)

		decoder_outputs_tgt, _ = self._build_decoder(decoder_inputs, [vec_tgt]*self.decoder_depth, layers)
		decoder_outputs_src, _ = self._build_decoder(decoder_inputs, [vec_src]*self.decoder_depth, layers)
		outputs += [decoder_outputs_tgt, decoder_outputs_src]

		# connections: interp

		if self.interp:
			vec_tgt_hat = Lambda(_interp, name='interp')([vec_src, vec_tgt])
			decoder_outputs_tgt_hat, _ = self._build_decoder(decoder_inputs, [vec_tgt_hat]*self.decoder_depth, layers)
			outputs.append(decoder_outputs_tgt_hat)

		# models
		self.model = Model(inputs, outputs)
		self._compile(loss_weights=self.loss_weights)
		self.plot_model()



	def build_model_test(self):
		self._build_model_test()
		layers = extract_layers(self.model)

		dial_encoder_inputs = Input(shape=(None,), name='dial_encoder_inputs')
		vec_res_hat_inp = Input(shape=(self.rnn_units,), name='vec_res_hat_inp')

		vec_src = self.model_encoder['dial'](dial_encoder_inputs)
		vec_tgt_hat = Add()([vec_src, vec_res_hat_inp])
		self.model_tgt = Model(
						[dial_encoder_inputs, vec_res_hat_inp],
						vec_tgt_hat,
						)



	def train_a_load(self, batch_size, batch_per_load):
		encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts = \
					self.dataset.load_data('train', batch_size * batch_per_load)

		n_sample = decoder_input_data.shape[0]
		if n_sample == 0:
			return 0

		t0 = datetime.datetime.now()
		t0_str = str(t0).split('.')[0]
		s = '\nstart:\t%s'%t0_str
		write_log(self.log_train, s)

		inputs = [
				encoder_input_data['auto'], 
				encoder_input_data['dial'], 
				decoder_input_data, 
				]
		outputs = [np.zeros((n_sample, self.rnn_units))] *3 + [decoder_target_data] * (2 + self.interp)

		self.model.fit(
			inputs, 
			outputs,
			batch_size=batch_size,
			callbacks=[self.history],
			verbose=FIT_VERBOSE)
		self.n_trained += n_sample

		dt = (datetime.datetime.now() - t0).seconds
		loss = np.mean(self.history.losses)

		# vali --------------------
		loss_vali = self.vali()

		# summary --------------------
		ss = [
			'spent: %i sec'%dt,
			'train: %.4f'%loss,
			'vali:  %.4f'%loss_vali[0],
			'-',
			'd(S,T):%.4f'%loss_vali[1],
			'd(S,S):%.4f'%loss_vali[2],
			'd(T,T):%.4f'%loss_vali[3],
			'auto:  %.4f'%loss_vali[4],
			'dial:  %.4f'%loss_vali[5],
			]
		if self.interp:
			ss.append('interp:%.4f'%loss_vali[-1])
		s = '\n'.join(ss)

		write_log(self.log_train, s)
		write_log(self.tsv_train, '\t'.join(map(strmap, 
			['%.1e'%self.lr, '%.3f'%(self.n_trained/1e6), t0_str, dt, loss] + loss_vali)), PRINT=False)

		if SKIP_VIS:
			return n_sample

		self.build_model_test()
		for i in range(5):
			s = '\n'.join([
				'-'*10,
				'source:    '+source_texts[i],
				'target:    '+target_texts[i],
				'auto_pred: '+self.dialog(target_texts[i], prefix='auto')[1],
				'dial_pred: '+self.dialog(source_texts[i], prefix='dial')[1],
				])
			write_log(self.log_train, s)

		return n_sample


	def vali(self):
		if self.vali_data is None:
			self._get_vali_data()

		encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts = self.vali_data
		inputs = [
				encoder_input_data['auto'], 
				encoder_input_data['dial'], 
				decoder_input_data, 
				]

		outputs = [np.zeros((decoder_input_data.shape[0], self.rnn_units))] *3 + [decoder_target_data] * (2 + self.interp)
		print('validating....')
		return self.model.evaluate(inputs, outputs, verbose=0)







class DecoderAR:
	"""
	autoregressive (run-time), greedy-search, decoder
	"""

	def __init__(self, dataset, model, decoder_depth):
		self.dataset = dataset
		self.model = model
		self.decoder_depth = decoder_depth

	def predict(self, latent, sampling=False):
		latent = np.atleast_2d(latent)
		prev = np.atleast_2d([self.dataset.SOS])
		states = [latent] * self.decoder_depth

		hyp = ''
		t = 0
		logP = 0.
		while True:

			out = self.model.predict([prev] + states)
			states = out[1:]
			tokens_proba = out[0].ravel()
			tokens_proba[self.dataset.UNK] = 0.	# avoid UNK
			tokens_proba = tokens_proba/sum(tokens_proba)

			if sampling:
				# soft-max sampling
				sampled_token_index = np.random.choice(
					range(len(tokens_proba)), 1, p=tokens_proba)[0]
			else:
				# greedy
				sampled_token_index = np.argmax(tokens_proba)

			logP += np.log(tokens_proba[sampled_token_index])
			sampled_token = self.dataset.index2token[sampled_token_index]
			hyp += sampled_token+' '
			t += 1
			if sampled_token_index == self.dataset.EOS or t > self.dataset.max_tgt_len:
				break
			prev = np.atleast_2d([sampled_token_index])

		return logP/t, hyp
