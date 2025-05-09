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
            'traffic': [1385.4058837890625, 859.8118896484375],
            'packets': [1.4015231132507324, 0.8932565450668335],
            'eq_lambda': [1350.97119140625, 858.316162109375],
            'avg_pkts_lambda': [0.9117304086685181, 0.9723503589630127],
            'exp_max_factor': [6.663637638092041, 4.715115070343018],
            'pkts_lambda_on': [0.9116322994232178, 1.651275396347046],
            'avg_t_off': [1.6649284362792969, 2.356407403945923],
            'avg_t_on': [1.6649284362792969, 2.356407403945923],
            'ar_a': [0.0, 1.0],
            'sigma': [0.0, 1.0],
            'capacity': [27611.091796875, 20090.62109375],
            'queue_size': [30259.10546875, 21410.095703125]
        }

        # BiLSTM Path Updates
        self.path_forward = tf.keras.layers.LSTMCell(self.path_state_dim // 2)
        self.path_backward = tf.keras.layers.LSTMCell(self.path_state_dim // 2)

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
        path_state = self.path_embedding(tf.concat([
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
        ], axis=1))

        path_state_h_forward = path_state[:, :self.path_state_dim // 2]
        path_state_c_forward = tf.zeros_like(path_state_h_forward)
        path_state_h_backward = path_state[:, self.path_state_dim // 2:]
        path_state_c_backward = tf.zeros_like(path_state_h_backward)

        link_state = self.link_embedding(tf.concat([load, policy], axis=1))
        link_state_h = link_state
        link_state_c = tf.zeros_like(link_state_h)

        queue_state = self.queue_embedding(tf.concat([
            (queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1],
            priority, weight
        ], axis=1))
        queue_state_h = queue_state
        queue_state_c = tf.zeros_like(queue_state_h)

        for it in range(self.iterations):
            queue_gather = tf.gather(queue_state_h, queue_to_path)
            link_gather = tf.gather(link_state_h, link_to_path)

            combined_input = tf.concat([queue_gather, link_gather], axis=2)

            previous_path_state_h = tf.concat([path_state_h_forward, path_state_h_backward], axis=-1)

            forward_rnn = tf.keras.layers.RNN(self.path_forward, return_sequences=True, return_state=True)
            backward_rnn = tf.keras.layers.RNN(self.path_backward, return_sequences=True, return_state=True, go_backwards=True)

            path_state_sequence_forward, path_state_h_forward, path_state_c_forward = forward_rnn(
                combined_input, initial_state=[path_state_h_forward, path_state_c_forward]
            )

            path_state_sequence_backward, path_state_h_backward, path_state_c_backward = backward_rnn(
                combined_input, initial_state=[path_state_h_backward, path_state_c_backward]
            )
            path_state_sequence_backward = tf.reverse(path_state_sequence_backward, axis=[1])

            path_state_sequence = tf.concat([path_state_sequence_forward, path_state_sequence_backward], axis=-1)

            path_state_sequence = tf.concat([tf.expand_dims(previous_path_state_h, 1), path_state_sequence], axis=1)

            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.reduce_sum(path_gather, axis=1)

            queue_state_output, [queue_state_h, queue_state_c] = self.queue_update(path_sum, [queue_state_h, queue_state_c])

            queue_gather = tf.gather(queue_state_h, queue_to_link)
            link_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False, return_state=True)
            link_state_output, link_state_h, link_state_c = link_rnn(queue_gather, initial_state=[link_state_h, link_state_c])

        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.reduce_sum(1 / capacity_gather, axis=1)

        return queue_delay + trans_delay
