package izzah.mp.covid.model;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class covidModel {

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Logger log = LoggerFactory.getLogger(covidModel.class);

    public static void main(String[] args) throws Exception{
        int height = 299;
        int width = 299;
        int channels =1;
        int outputNum = 3;
        int batchSize = 128;
        int nEpochs = 10;
        int seed = 1234;

        Random rand = new Random(seed);
        log.info("Data vectorization...");

        // --------------Train Data Preparation----------------
        // set parent directory for train data that have done augmentation process
        File parentDir = new ClassPathResource("/train").getFile();
        FileSplit trainSplit = new FileSplit(parentDir, allowedExtensions,rand);

        // set label maker based on the subfolder name ex. Inside train consist of COVID, Normal, Viral Pneumonia
        // 3 labels will be generate that are 'COVID', 'Normal', 'Viral Pneumonia'
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader img_train = new ImageRecordReader(height,width,channels,labelMaker);
        img_train.initialize(trainSplit);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(img_train,batchSize,1,outputNum);

        // --------------Test Data Preparation---------------------
        // set parent directory for test data that are not go through augmentation process
        File testData = new ClassPathResource("/test").getFile();
        FileSplit testSplit = new FileSplit(testData,allowedExtensions,rand);

        // set label maker based on the subfolder name ex. Inside test consist of COVID, Normal, Viral Pneumonia
        // 3 labels will be generate that are 'COVID', 'Normal', 'Viral Pneumonia'
        ParentPathLabelGenerator testLM = new ParentPathLabelGenerator();
        ImageRecordReader img_test = new ImageRecordReader(height,width,channels,testLM);
        img_test.initialize(testSplit);

        DataSetIterator testIter = new RecordReaderDataSetIterator(img_test,batchSize,1,outputNum);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(Adam.DEFAULT_ADAM_LEARNING_RATE))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(1e-4)
                .list()
                .layer(new ConvolutionLayer.Builder(11,11)
                        .nIn(channels)
                        .stride(4,4)
                        .activation(Activation.RELU)
                        .nOut(256)
                        //.padding(1,1)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(3,3)
                        .padding(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder(5,5)
                        .nOut(384)
                        .stride(1,1)
                        .padding(2,2)
                        .activation(Activation.RELU)
                        .biasInit(0.1)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .stride(1,1)
                        .kernelSize(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3,3)
                        .nOut(384)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .biasInit(0.1)
                        .dropOut(0.2)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .stride(2,2)
                        .kernelSize(3,3)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(600)
                        .biasInit(0.1)
                        .dropOut(0.5)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(outputNum)
                        .build())
                .setInputType(InputType.convolutionalFlat(height,width,channels))
                .backpropType(BackpropType.Standard)
                .build();

        // define network
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info(model.summary());

        // set UI-Evaluator
//        StatsStorage storage = new InMemoryStatsStorage();
//        UIServer server = UIServer.getInstance();
//        server.attach(storage);

        // set model listeners
//        model.setListeners(new StatsListener(storage,10));
        model.setListeners(new ScoreIterationListener(10));

        // save model
        String folder = "E:\2021\covid19Diagnosis\src\main\resources";
        File locationToSave = new File(folder,"/trained_covid_model.zip");
        boolean saveUpdater = true;
        ModelSerializer.writeModel(model,locationToSave,saveUpdater);

        // training started and test with test dataset
        for(int i=0; i<nEpochs; i++){
            model.fit(trainIter);
            log.info("Completed epoch at " + i);
            Evaluation eval = model.evaluate(testIter);
            log.info(eval.stats());

        }
        Evaluation eTrain = model.evaluate(trainIter);
        Evaluation eTest = model.evaluate(testIter);
        log.info("Train Data");
        log.info(eTrain.stats());

        log.info("Test Data");
        log.info(eTest.stats());



    }
}
