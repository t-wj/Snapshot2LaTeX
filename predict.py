import logging
import os
import numpy as np
from easydict import EasyDict as edict
from skimage.filters import threshold_otsu
from scipy import ndimage
from models.seq2seq_model import Seq2SeqAttModel

class ConfigSeq2Seq:
    def __init__(self, data_type, gpu, encoder_type='conv'):
        self._data_type = data_type
        self.gpu = gpu
        self.encoder_type = encoder_type
        self._configs = edict()
        self._configs.datatype = self.encoder_type
        self.model()
        self.dataset()
        self.predict()
        

    def model(self):
        
        _model = edict()
        _model.batch_size = 16
        _model.test_batch_size = 1
        _model.gpu_flage = self.gpu >= 0
        _model.gpu_fraction = 0.7
        # _model.optimizer = 'momentum'
        # _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        _model.learning_decay_step = 8000
        _model.learning_decay_rate = 0.94
        _model.encoder_name = 'Encode'
        # The different between Augment and conv:
        #  Augment: image enhance and adadalta optimizer
        #  conv: image normal and momentum optmizer
        if self.encoder_type == 'Augment':
            # _model.encoder_type = 'Augment'  # ['Augment','conv']
            # _model.learning_init = 0.1
            # _model.optimizer = 'adadelta'
            # _model.learning_type = 'fixed'  # ['exponential','fixed','polynomial']
            _model.encoder_type = 'Augment'
            _model.learning_init = 0.001
            _model.optimizer = 'momentum'
            _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        else:
            _model.encoder_type = 'conv'
            _model.learning_init = 0.001
            _model.optimizer = 'momentum'
            _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        _model.decoder_name = 'DecoderAtt'
        _model.encoder_cnn = "vanilla"
        _model.droupout = 0.3  # droupout rate
        _model.positional_embeddings = True
        _model.rnn_encoder_dim = 256  # rnn encoder num unit
        _model.embeding_dims = 80  # word embeding dimision
        _model.rnn_decoder_dim = 512  # rnn decoder num unit
        _model.att_dim = 512
        _model.clip_value = 5
        _model.save_iter = 500
        _model.display_iter = 100
        _model.beam_size = 5
        _model.div_gamma = 1
        _model.div_prob = 0
        _model.n_epochs = 1000
        _model.MaxPredictLength = 200
        _model.decoding = 'beams_search'  # chose from ['greedy','beams_search']
        # _model.decoding='greedy'
        _model.metric_val = 'perplexity'
        _model.ckpt_name = 'seq2seqAtt'
        _model.ckpt_dir = './checkpoint'

        _model.log_dir = os.path.abspath('./log')  # log path
        _model.log_name = 'Im2Katex'
        _model.log_file_name = 'Im2Katex.log'
        self._configs.model = _model

    def dataset(self):
        _dataset = edict()
        _dataset.id_start = 0
        _dataset.id_end = 1
        _dataset.id_unk = 2
        _dataset.id_pad = 3
        _dataset.vocabulary_file = './properties.npy'

        self._configs.dataset = _dataset

    def predict(self):
        """ 
        The predict details want to be displayed o web, 
        so the root image is the "static" which is the flask defaulet static folder 
        """
        _predict = edict()
        _predict.web_path = './templates'
        # root dir
        _predict.temp_path = './static'
        # preprocess folder for the predict
        _predict.preprocess_dir = os.path.join(_predict.temp_path, 'preprocess')
        # save details on the numpy format
        _predict.npy_path = os.path.join(_predict.temp_path, 'npy')
        # # if the input is an pdf, the convert it
        # _predict.pdf_path = os.path.join(_predict.preprocess, 'pdf')
        # # crop the input image
        # _predict.croped_path = os.path.join(_predict.preprocess, 'croped')
        # # resize the input image
        # _predict.resized_path = os.path.join(_predict.preprocess, 'resized')
        # # pad the input image
        # _predict.pad_path = os.path.join(_predict.preprocess, 'pad')
        # render the image based on latex predicted by the given image
        _predict.render_path = os.path.join(_predict.temp_path, 'render')
        # # crop the rendered image and save it
        # _predict.render_out_path = os.path.join(_predict.temp_path, 'render', 'out')

        self._configs.predict = _predict

class VocabSeq2Seq:
    def __init__(self, config, logger,vacab_file=None):
        self._config = config
        self._logger = logger
        self.vacab_file=vacab_file
        self.load_vocab()

    def load_vocab(self):
        if self.vacab_file is None:
            vocab_dir = os.path.abspath(self._config.dataset.vocabulary_file)
        else:
            vocab_dir=self.vacab_file
        print("vocab_dir",vocab_dir)
        vocabulary = np.load(vocab_dir, allow_pickle=True).tolist()
        self.vocab_size = vocabulary['vocab_size']
        self.idx_to_token = vocabulary['idx_to_str']
        self.token_to_idx = vocabulary['str_to_idx']
        self.bucket_size = [(687, 24), (598, 24), (597, 32), (450, 32),
                            (569, 64), (762, 48), (703, 64), (256, 32),
                            (591, 40), (525, 40), (335, 40), (593, 48), (152, 48),
                            (505, 64), (311, 64), (381, 32), (197, 32), (398, 40),
                            (83, 40), (376, 64), (245, 64), (199, 24), (738, 40),
                            (140, 32), (678, 32), (676, 48), (441, 64), (351, 24),
                            (636, 64), (126, 24), (147, 40), (777, 24), (512, 24),
                            (512, 48), (660, 40), (218, 48), (359, 48), (778, 64),
                            (461, 40), (274, 24), (272, 40), (287, 48), (317, 32),
                            (210, 40), (522, 32), (178, 64), (430, 24), (434, 48)]
        self.target_height = list(set(idx[1] for idx in self.bucket_size))
        self._logger.info('Vocab size is [{:d}]'.format(self.vocab_size))

class Predict:
    def __init__(self):
        _dataset_type = 'merged'
        _gpu = -1
        _encoder_type = 'conv'
        _Configure = ConfigSeq2Seq(_dataset_type, _gpu, _encoder_type)
        # Get configures for the project
        _config = _Configure._configs

        _loggerDir = _config.model.log_dir
        log_path = _config.model.log_file_name
        logger_name = _config.model.log_name
        _LogFile = os.path.join(_loggerDir, log_path)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        # ccreate file handler which logs even debug messages
        fh = logging.FileHandler(_LogFile)
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        _LogFormat = logging.Formatter("%(asctime)2s -%(name)-12s:  %(levelname)-10s - %(message)s")

        fh.setFormatter(_LogFormat)
        console.setFormatter(_LogFormat)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(console)

        logger.info('Logging is working ...')
        # Generate the vocab
        _vocab = VocabSeq2Seq(_config, logger)

        target_height = _vocab.target_height
        bucket_size = _vocab.bucket_size
        self.Model = Seq2SeqAttModel(config=_config, vocab=_vocab, logger=logger, trainable=False)
        self.Model.build_inference()
        _ = self.Model.restore_session()

        self.half_max_height = 36

    def predict_img(self, img_raw):
        thr = min(245, threshold_otsu(img_raw) + 20)
        img_raw = (img_raw > thr) * 255

        nnz_inds = np.where(img_raw != 255)
        if len(nnz_inds[0]) == 0:
            return ''
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        if (x_max - x_min) * (y_max - y_min) < 100:
            return ''
        img_raw = img_raw[y_min:(y_max+1), x_min:(x_max+1)]

        zoom = 2 ** int(np.log2(np.abs(img_raw.shape[0] / self.half_max_height)))
        img_raw = ndimage.zoom(img_raw, 1.0 / zoom)
        img_raw = (img_raw > thr) * 255
        _predict_latex_list = self.Model.predict_single_img(img_raw[:, :, np.newaxis])
        return _predict_latex_list[0]
