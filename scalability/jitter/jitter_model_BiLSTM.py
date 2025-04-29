import tensorflow as tf

class RouteNet_Fermi(tf.keras.Model):
    def __init__(self):
        super(RouteNet_Fermi, self).__init__()

        self.max_num_models = 7
        self.num_policies = 4
        self.max_num_queues = 3

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.z_score = {
            'traffic': [1385.4059, 859.8119],
            'packets': [1.4015, 0.8933],
            'eq_lambda': [1350.9712, 858.3162],
            'avg_pkts_lambda': [0.9117, 0.9724],
            'exp_max_factor': [6.6636, 4.7151],
            'pkts_lambda_on': [0.9116, 1.6513],
            'avg_t_off': [1.6649, 2.3564],
            'avg_t_on': [1.6649, 2.3564],
            'ar_a': [0.0, 1.0],
            'sigma': [0.0, 1.0],
            'capacity': [27611.0918, 20090.6211],
            'queue_size': [30259.1055, 21410.0957]
        }

        # BiLSTM: forward and backward path updates
        self.path_update_fw = tf.keras.layers.LSTMCell(self.path_state_dim // 2)
        self.path_update_bw = tf.keras.layers.LSTMCell(self.path_state_dim // 2)

        self.link_update = tf.keras.layers.LSTMCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.LSTMCell(self.queue_state_dim)

        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10 + self.max_num_models,)),
            tf.keras.layers.Dense(self.path_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.path_state_dim, activation='relu')
        ])

        self.queue_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_num_queues + 2,)),
            tf.keras.layers.Dense(self.queue_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.queue_state_dim, activation='relu')
        ])

        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.num_policies + 1,)),
            tf.keras.layers.Dense(self.link_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.link_state_dim, activation='relu')
        ])

        self.readout_path = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            tf.keras.layers.Dense(self.link_state_dim // 2, activation='relu'),
            tf.keras.layers.Dense(self.path_state_dim // 2, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name="PathReadout")

    @tf.function
    def call(self, inputs):
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        model = inputs['model']
        eq_lambda = inputs['eq_lambda']
        avg_pkts_lambda = inputs['avg_pkts_lambda']
        exp_max_factor = inputs['exp_max_factor']
        pkts_lambda_on = inputs['pkts_lambda_on']
        avg_t_off = inputs['avg_t_off']
        avg_t_on = inputs['avg_t_on']
        ar_a = inputs['ar_a']
        sigma = inputs['sigma']

        capacity = inputs['capacity']
        policy = tf.one_hot(inputs['policy'], self.num_policies)

        queue_size = inputs['queue_size']
        priority = tf.one_hot(inputs['priority'], self.max_num_queues)
        weight = inputs['weight']

        queue_to_path = inputs['queue_to_path']
        link_to_path = inputs['link_to_path']
        path_to_link = inputs['path_to_link']
        path_to_queue = inputs['path_to_queue']
        queue_to_link = inputs['queue_to_link']

        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

        pkt_size = traffic / packets

        # Embedding
        path_input = tf.concat([
            (traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
            (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
            tf.one_hot(model, self.max_num_models),
            (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
            (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
            (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
            (pkts_lambda_on - self.z_score['pkts_lambda_on'][0]) / self.z_score['pkts_lambda_on'][1],
            (avg_t_off - self.z_score['avg_t_off'][0]) / self.z_score['avg_t_off'][1],
            (avg_t_on - self.z_score['avg_t_on'][0]) / self.z_score['avg_t_on'][1],
            (ar_a - self.z_score['ar_a'][0]) / self.z_score['ar_a'][1],
            (sigma - self.z_score['sigma'][0]) / self.z_score['sigma'][1]
        ], axis=1)

        path_state = self.path_embedding(path_input)
        path_h_fw = path_state[:, :self.path_state_dim // 2]
        path_c_fw = tf.zeros_like(path_h_fw)
        path_h_bw = path_state[:, self.path_state_dim // 2:]
        path_c_bw = tf.zeros_like(path_h_bw)

        link_state = self.link_embedding(tf.concat([load, policy], axis=1))
        link_h = link_state
        link_c = tf.zeros_like(link_state)

        queue_state = self.queue_embedding(
            tf.concat([(queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1], priority, weight], axis=1))
        queue_h = queue_state
        queue_c = tf.zeros_like(queue_state)

        for it in range(self.iterations):
            queue_gather = tf.gather(queue_h, queue_to_path)
            link_gather = tf.gather(link_h, link_to_path)

            combined_input = tf.concat([queue_gather, link_gather], axis=2)

            prev_path_h = tf.concat([path_h_fw, path_h_bw], axis=-1)

            forward_rnn = tf.keras.layers.RNN(self.path_update_fw, return_sequences=True, return_state=True)
            backward_rnn = tf.keras.layers.RNN(self.path_update_bw, return_sequences=True, return_state=True, go_backwards=True)

            path_seq_fw, path_h_fw, path_c_fw = forward_rnn(combined_input, initial_state=[path_h_fw, path_c_fw])
            path_seq_bw, path_h_bw, path_c_bw = backward_rnn(combined_input, initial_state=[path_h_bw, path_c_bw])
            path_seq_bw = tf.reverse(path_seq_bw, axis=[1])

            path_seq = tf.concat([path_seq_fw, path_seq_bw], axis=-1)
            path_seq = tf.concat([tf.expand_dims(prev_path_h, 1), path_seq], axis=1)

            path_gather = tf.gather_nd(path_seq, path_to_queue)
            path_sum = tf.reduce_sum(path_gather, axis=1)

            queue_state_output, [queue_h, queue_c] = self.queue_update(path_sum, [queue_h, queue_c])

            queue_gather = tf.gather(queue_h, queue_to_link)
            link_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False, return_state=True)
            link_state_output, link_h, link_c = link_rnn(queue_gather, initial_state=[link_h, link_c])

        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_seq[:, 1:].to_tensor()


        ratio_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        ratio_gather = tf.RaggedTensor.from_tensor(ratio_gather, lengths=length)

        jitter = tf.math.reduce_sum(ratio_gather / capacity_gather, axis=1)
        return jitter
