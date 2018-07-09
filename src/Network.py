import numpy as np
import tensorflow as tf
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Network:
    def __init__(self, saveDir, networkConstructor=None, tensorflowConfig={}):

        self.batchCount = 0

        self.graph = tf.Graph()
        gpuOptions = tensorflowConfig.get(
            'GPUOptions', {'definitelyNotAKey': None})
        gpuOptions = tf.GPUOptions(**gpuOptions)
        self.sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpuOptions), graph=self.graph)

        with self.graph.as_default():
            with self.sess.as_default():
                networkConstructor()
                self.grabVariables()
                self.sess.run(tf.global_variables_initializer())
                self.saveModel(saveDir)

    def __del__(self):
        self.sess.close()

    def grabVariables(self):
        self.input = tf.get_collection('input')[0]
        self.evaluation = tf.get_collection('evaluation')[0]
        self.policy = tf.get_collection('policy')[0]
        self.evaluationLabel = tf.get_collection('evaluationLabel')[0]
        self.policyLabel = tf.get_collection('policyLabel')[0]
        self.learningRate = tf.get_collection('learningRate')[0]
        self.trainingOp = tf.get_collection('trainingOp')[0]
        self.lossMerged = tf.get_collection('lossMerged')[0]

        if len(tf.get_collection('teacherPolicy')) > 0:
            self.teacherPolicy = tf.get_collection('teacherPolicy')[0]

    def getEvaluation(self, state):
        """ Given a game state, return the network's evaluation.
        """
        evaluation = self.sess.run(
            self.evaluation, feed_dict={self.input: state})

        return evaluation[0]

    def getPolicy(self, state):
        """ Given a game state, return the network's policy.
            Random Dirichlet noise is applied to the policy output
            to ensure exploration, if training.
        """
        policy = self.sess.run(
            self.policy, feed_dict={self.input: state})

        return policy[0]

    def train(self, state, eval, policy, learningRate=0.01, teacher=None):
        """ Train the network
        """
        feed_dict = {
            self.input: state,
            self.evaluationLabel: eval,
            self.policyLabel: policy,
            self.learningRate: learningRate
        }

        if teacher is not None:
            feed_dict[self.teacherPolicy] = teacher.getPolicy(state)

        self.sess.run(self.trainingOp, feed_dict=feed_dict)
        self.batchCount += 1
        if self.batchCount % 10 == 0:
            # TODO : Log this data in a way that we can actually use it.
            summary = self.sess.run(self.lossMerged, feed_dict=feed_dict)

    def saveModel(self, saveDir):
        """ Write the state of the network to a file.
            This should be reserved for "best" networks.
        """
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        with self.sess.graph.as_default():
            tf.train.Saver().save(self.sess, os.path.join(saveDir, 'best'))
