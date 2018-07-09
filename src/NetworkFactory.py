import tensorflow as tf
import os


class NetworkFactory(object):
    """ Class which allows for simpler limited construction of a neural network from a few parameters.
        Args:
            `networkConfig`: Not goina lie. You probably shouldn't be trying to use any of this.
    """

    def __init__(self, networkConfig):
        self.NetworkConfig = networkConfig
        if isinstance(networkConfig, dict):
            self.alpha = networkConfig.get('policy').get('dirichlet').get('alpha')
            self.epsilon = networkConfig.get(
                'policy').get('dirichlet').get('epsilon')
            self.dims = networkConfig.get('dims')
            self.policyShape = networkConfig.get('policyShape')
            self.hasTeacher = networkConfig.get('hasTeacher')
            setattr(self, '__call__', self.buildNew)


    def __call__(self):
        if isinstance(self.NetworkConfig, dict):
            self.buildNew()
        else:
            self.loadFromFile()

    def loadFromFile(self):
        meta = os.path.join(self.NetworkConfig, 'best.meta')
        saver = tf.train.import_meta_graph(meta, clear_devices=True)
        saver.restore(tf.get_default_session(), os.path.join(self.NetworkConfig, 'best'))


    def buildNew(self):
        """ Build out the policy/evaluation combo network
        """
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            input = tf.placeholder(
                shape=[None] + self.dims + [3],
                name='board_input', dtype=tf.float32)
            tf.add_to_collection('input', input)

            policyLabel = tf.placeholder(
                shape=[None, self.policyShape],
                name='correct_move_from_mcts', dtype=tf.float32)
            tf.add_to_collection('policyLabel', policyLabel)

            evaluationLabel = tf.placeholder(
                shape=[None], name='evaluationLabel', dtype=tf.float32)
            tf.add_to_collection('evaluationLabel', evaluationLabel)

        with tf.variable_scope('resTower', reuse=tf.AUTO_REUSE):
            resTower = [input]

            with tf.variable_scope('conv_block', reuse=tf.AUTO_REUSE):
                """ AlphaZero convolutional blocks are...
                    1) Convolutional layer of 256 3x3 filters, stride of 1
                    2) Batch normalization
                    3) Rectifier nonlinearity
                """
                resTower.append(
                    tf.layers.conv2d(inputs=input, kernel_size=[3, 3],
                                     strides=1, padding="same", name='conv',
                                     filters=self.NetworkConfig['filters']))

                resTower.append(
                    tf.layers.batch_normalization(
                        inputs=resTower[-1], name='batch_norm'))

                resTower.append(
                    tf.nn.relu(
                        features=resTower[-1], name='rect_nonlinearity'))

            for block in range(self.NetworkConfig['blocks']):
                with tf.variable_scope('block_{}'.format(block), reuse=tf.AUTO_REUSE):
                    """ AlphaZero residual blocks are...
                        1) Convolutional layer of 256 3x3 filters, stride of 1
                        2) Batch normalization
                        3) Rectifier nonlinearity
                        4) Convolutional layer of 256 3x3 filters, stride of 1
                        5) Bath normalization
                        6) Skip connection that adds the input to the block
                        7) Rectifier nonlinearity
                    """
                    resTower.append(
                        tf.layers.conv2d(
                            inputs=resTower[-1], kernel_size=[3, 3],
                            strides=1, filters=self.NetworkConfig['filters'],
                            padding="same", name='conv_1'))

                    resTower.append(
                        tf.layers.batch_normalization(
                            inputs=resTower[-1], name='batch_norm_1'))

                    resTower.append(
                        tf.nn.relu(
                            features=resTower[-1],
                            name='rectifier_nonlinearity_1'))

                    resTower.append(
                        tf.layers.conv2d(
                            inputs=resTower[-1], kernel_size=[3, 3],
                            filters=self.NetworkConfig['filters'], strides=1,
                            padding="same", name='conv_2'))

                    resTower.append(
                        tf.layers.batch_normalization(
                            inputs=resTower[-1], name='batch_norm_2'))

                    resTower.append(
                        tf.add(
                            resTower[-1], resTower[-6],
                            name='skip_connection'))

                    resTower.append(
                        tf.nn.relu(
                            features=resTower[-1],
                            name='rectifier_nonlinearity_2'))

        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            """ AlphaZero's value head is...
                1) Convolutional layer of 2 1x1 filters, stride of 1
                2) Batch normalization
                3) Rectifier nonlinearity
                4) Fully connected layer of size 256
                5) Rectifier nonlinearity
                6) Fully connected layer of size 1
                7) tanh activation
            """
            evalConv = tf.layers.conv2d(
                resTower[-1], filters=1, kernel_size=(1, 1), strides=1,
                name='convolution')

            evalBatchNorm = tf.layers.batch_normalization(
                evalConv, name='batch_norm')

            evalRectifier1 = tf.nn.relu(
                evalBatchNorm, name='rect_norm_1')

            evalDense = tf.layers.dense(
                inputs=evalRectifier1,
                units=self.NetworkConfig.get('eval').get('dense'), name='dense_1')

            evalDenseReduced = tf.reduce_sum(
                evalDense,
                axis=[1, 2], name='reduced_dense')

            evalRectifier2 = tf.nn.relu(
                evalDenseReduced, name='rect_norm_2')

            evalDenseScalar = tf.layers.dense(
                inputs=evalRectifier2,
                units=1, name='dense_2')

            evalScalar = tf.reduce_sum(
                evalDenseScalar,
                axis=[1], name='reduced_scalar')

            evaluation = tf.tanh(
                evalScalar, name='value')
            tf.add_to_collection('evaluation', evaluation)

        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
            """ AlphaZero's policy head is...
                1) Convolutional layer of 2 1x1 filters, stride of 1
                2) Batch normalization
                3) Rectifier nonlinearity
                4) Fully connected layer of size |legal actions|
            """
            policyConv = tf.layers.conv2d(
                resTower[-1], filters=2, kernel_size=(1, 1), strides=1,
                name='convolution')

            policyBatchNorm = tf.layers.batch_normalization(
                policyConv, name='batch_norm')

            policyRectifier = tf.nn.relu(
                policyBatchNorm, name='rect_norm')

            policyDense = tf.layers.dense(
                policyRectifier, units=self.policyShape, name='policy')

            policyVector = tf.reduce_sum(
                policyDense, axis=[1, 2])

            policyBase = tf.nn.softmax(
                policyVector)

            # Generate Dirichlet noise to add to the network policy

            dist = tf.distributions.Dirichlet(
                [self.alpha, 1-self.alpha])

            policy = ((1 - self.epsilon) * policyBase
                      + self.epsilon * dist.sample([1, self.policyShape])[0][:, 0])

            policy /= tf.reduce_sum(policy)
            tf.add_to_collection('policy', policy)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):

            lossEvaluation = tf.reduce_mean(tf.square(
                evaluation - evaluationLabel))

            lossPolicy = -tf.reduce_mean(
                tf.tensordot(
                    tf.log(policy),
                    tf.transpose(policyLabel),
                    axes=1))

            lossParam = tf.reduce_mean([
                tf.nn.l2_loss(v) for v in tf.trainable_variables()

                # I don't know if this filter is a good idea...
                if 'bias' not in v.name
            ])

            loss = lossEvaluation + lossPolicy + lossParam

            if self.hasTeacher:
                teacherPolicy = tf.placeholder(
                    shape=[policy.shape[1]], dtype=tf.float32,
                    name='teacher_policy')
                tf.add_to_collection('teacherPolicy', teacherPolicy)
                policyXentropy = -tf.reduce_mean(
                    tf.tensordot(
                        tf.log(teacherPolicy),
                        tf.transpose(policy),
                        axes=1),
                    axis=1)

                loss += policyXentropy
            tf.add_to_collection('loss', loss)

            avgLoss = tf.summary.scalar('average_loss', loss)
            policyLoss = tf.summary.scalar('policyLoss', lossPolicy)
            evalLoss = tf.summary.scalar('evalLoss', lossEvaluation)
            l2Loss = tf.summary.scalar('l2Loss', lossParam)

            lossMerged = tf.summary.merge([avgLoss, policyLoss,
                                           evalLoss, l2Loss])
            tf.add_to_collection('lossMerged', lossMerged)

        with tf.variable_scope('training', reuse=tf.AUTO_REUSE):
            learningRate = tf.placeholder(
                shape=(), dtype=tf.float32, name='learningRate')
            tf.add_to_collection('learningRate', learningRate)

            if self.NetworkConfig['training']['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(learningRate)
            elif self.NetworkConfig['training']['optimizer'] == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learningRate,
                    momentum=self.NetworkConfig['training']['momentum'])
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learningRate)

            trainingOp = optimizer.minimize(loss)
            tf.add_to_collection('trainingOp', trainingOp)
        return
