package izzah.mp.covid.ui;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class main extends Application {

    Stage window;
    BorderPane layout,layout2;
    Scene scene,scene2;
    TextField resultTxt = new TextField();
    File filePath;

    private static Logger log = LoggerFactory.getLogger(main.class);
    @Override
    public void start(Stage stage) throws Exception {
        // Assigning stage variable as window and set the tile for the window
        window = stage;
        window.setTitle("Covid 19 Application");

        //Initialize the custom font that were used for the application
        Font.loadFont(main.class.getResource("/Timeline.ttf").toExternalForm(),10);

        covidXray imgNew = new covidXray();

        //----------Borderpane layout for first scene-------------
        //-----------> Configuration for Top layout <-------------

        // VBox --> Menu, Label
        VBox vbox = new VBox(10);
        Menu fileMenu = new Menu("File");

        // MenuItem for fileMenu
        MenuItem newFile = new MenuItem("New");
        newFile.setOnAction(e -> System.out.println("Create new file"));
        MenuItem openFile = new MenuItem("Open");
        MenuItem settingFile = new MenuItem("Setting");
        MenuItem exit = new MenuItem("Exit");
        exit.setOnAction(e->{
            window.close();
        });
        fileMenu.getItems().addAll(newFile,openFile,settingFile,exit);

        //  Edit Menu
        Menu editMenu = new Menu("Edit");
        editMenu.getItems().add(new MenuItem("Cut"));
        editMenu.getItems().add(new MenuItem("Copy"));

        // Help Menu
        Menu helpMenu = new Menu("Help");

        //Main menu bar
        MenuBar menuBar = new MenuBar();
        menuBar.getMenus().addAll(fileMenu,editMenu,helpMenu);
        menuBar.setId("menubar-style");

        Label appLabel = new Label("COVID_19 APPLICATION");
        appLabel.setId("label-app");

        vbox.getChildren().addAll(menuBar,appLabel);
        vbox.setAlignment(Pos.CENTER);


        //-----------> Configuration for Left layout <-------------

        // AnchorPane as base of the left
        AnchorPane anchorPane = new AnchorPane();
        anchorPane.setId("pane");

        // ImageView
        Image img = new Image("/white.png");
        ImageView imgView = new ImageView(img);
        imgView.setFitWidth(250);
        imgView.setFitHeight(250);
        imgView.setX(30);
        imgView.setY(20);
        imgView.setPreserveRatio(true);
        imgView.setSmooth(true);

        // TextArea
        TextArea description = new TextArea();
        description.setText("Coronavirus disease (COVID-19) is an infectious\n" +"disease caused by a newly discovered coronavirus.\n" +
                "Most people infected with the COVID-19 virus\n"+"will experience mild to moderate respiratory illness\n" +
                "and recover without requiring special treatment.");
        description.setId("desc-label");
        description.setEditable(false);
        description.setPrefHeight(150);
        description.setPrefWidth(300);
        description.setLayoutX(15);
        description.setLayoutY(240);

        anchorPane.getChildren().addAll(imgView,description);

        //-----------> Configuration for Right layout <-------------

        // AnchorPane as the base
        AnchorPane rightAP = new AnchorPane();
        rightAP.setId("pane");

        // ImageView
        Image tmpImg = new Image("/xray2.png");
        ImageView temp = new ImageView(tmpImg);
        temp.setId("img-view");
        temp.setLayoutX(28);
        temp.setLayoutY(28);
        temp.setFitWidth(200);
        temp.setFitHeight(200);
        temp.setSmooth(true);
        temp.setPreserveRatio(true);

        // Upload Button
        Button btnUpload = new Button("Upload");
        btnUpload.setLayoutX(30);
        btnUpload.setLayoutY(290);
        btnUpload.setOnAction(e ->{
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Open..");

            FileChooser.ExtensionFilter extJPG = new FileChooser.ExtensionFilter("JPG type","*.jpg");

            filePath  = fileChooser.showOpenDialog(window);
            if(filePath != null)
            {
                // Desktop desktop = Desktop.getDesktop();
                try {
                    //desktop.open(file);
                    BufferedImage bufferedImage = ImageIO.read(filePath);
                    BufferedImage bf = ImageIO.read(filePath);
                    Image newImage = SwingFXUtils.toFXImage(bufferedImage,null);
                    imgNew.setXray(newImage);
                    temp.setImage(imgNew.getXray());
                    ImageIO.write(SwingFXUtils.fromFXImage(temp.getImage(),null),"png",filePath);

                    NativeImageLoader loader = new NativeImageLoader(299,299,1);
                    INDArray im = loader.asMatrix(filePath);
                    System.out.println(im);

                } catch (IOException ioException) {
                    ioException.printStackTrace();
                }
            }
        });

        // Predict Button
        Button btnPredict = new Button("Predict");
        btnPredict.setLayoutX(190);
        btnPredict.setLayoutY(290);
        btnPredict.setOnAction(e->{
            window.setScene(scene2);
            try {
                loadCovid19Model(resultTxt); // to test model exist or not
//                ImageView t = new ImageView(temp.getImage());
//                AnchorPane r = new AnchorPane();
//                r.getChildren().add(t);
//                Scene showImg = new Scene(r,600,600);
//                window.setScene(showImg);
//                window.show();

                //INDArray image = loader.asMatrix(temp.getImage());
                testImage(filePath,resultTxt); // predict the image and produce the result
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        });
        rightAP.getChildren().addAll(temp,btnUpload,btnPredict);


        //-----------> Configuration for Bottom layout <-------------

        // HBox as the base
        HBox bottom = new HBox();
        Label btmLabel = new Label("2021");
        btmLabel.setId("btm-label");
        bottom.getChildren().addAll(btmLabel);
        bottom.setPadding(new Insets(2,2,2,2));
        bottom.setAlignment(Pos.CENTER);

        //------------> set all the node Top, Left, Right, Bottom to the BorderPane <----------------------
        layout = new BorderPane();
        layout.setTop(vbox);
        layout.setLeft(anchorPane);
        layout.setBottom(bottom);
        layout.setCenter(rightAP);

        //------------> Main/first scene setup <---------------
        scene = new Scene(layout,600,550);
        scene.getStylesheets().add("/style.css");
        window.setScene(scene);
        window.show();

        //----------Borderpane layout for second scene-------------
        //-----------> Configuration for Top layout <-------------
        // File Menu
        VBox vbox2 = new VBox(10);
        Menu fileMenu2 = new Menu("File");

        // Menu File
        MenuItem newFile2 = new MenuItem("New");
        newFile2.setOnAction(e -> System.out.println("Create new file"));

        MenuItem openFile2 = new MenuItem("Open");
        MenuItem settingFile2 = new MenuItem("Setting");
        MenuItem exit2 = new MenuItem("Exit");
        exit2.setOnAction(e->{
            window.close();
        });
        fileMenu2.getItems().addAll(newFile2,openFile2,settingFile2,exit2);

        // Edit Menu
        Menu editMenu2 = new Menu("Edit");
        editMenu2.getItems().add(new MenuItem("Cut"));
        editMenu2.getItems().add(new MenuItem("Copy"));

        // Help Menu
        Menu helpMenu2 = new Menu("Help");

        // Main menu bar
        MenuBar menuBar2 = new MenuBar();
        menuBar2.getMenus().addAll(fileMenu2,editMenu2,helpMenu2);
        menuBar2.setId("menubar-style");

        Label result = new Label("RESULT");
        result.setId("label-app");

        vbox2.getChildren().addAll(menuBar2,result);
        vbox2.setAlignment(Pos.CENTER);

        // Bottom scene 2
        HBox bottom2 = new HBox();
        Label btmLabel2 = new Label("2021");
        btmLabel2.setId("btm-label");
        bottom2.getChildren().addAll(btmLabel2);
        bottom2.setPadding(new Insets(2,2,2,2));
        bottom2.setAlignment(Pos.CENTER);

        // Center scene2
        VBox vcenter = new VBox();

        resultTxt.setAlignment(Pos.TOP_LEFT);
        resultTxt.setPrefSize(100,150);
        resultTxt.setEditable(false);
        resultTxt.appendText("\n");

        Region r2 = new Region();
        r2.setPrefSize(100,25);

        HBox hcenter =new HBox();
        hcenter.setPrefSize(295,34);
        Button btnBack = new Button("Back");
        btnBack.setOnAction(e-> window.setScene(scene));

        Region region = new Region();
        region.setPrefSize(100,25);

        Button btnClose = new Button("Close");
        btnClose.setOnAction(e-> window.close());

        hcenter.getChildren().addAll(btnBack,region,btnClose);
        hcenter.setAlignment(Pos.CENTER);
        vcenter.getChildren().addAll(resultTxt,r2,hcenter);

        AnchorPane apL = new AnchorPane();
        Region r3 = new Region();
        r3.setPrefWidth(100);
        apL.getChildren().add(r3);

        AnchorPane apR = new AnchorPane();
        Region r4 = new Region();
        r4.setPrefWidth(100);
        apR.getChildren().add(r4);

        //------------> set all the node Top, Left, Right, Bottom to the BorderPane <----------------------
        layout2 = new BorderPane();
        layout2.setTop(vbox2);
        layout2.setBottom(bottom2);
        layout2.setCenter(vcenter);
        layout2.setLeft(apL);
        layout2.setRight(apR);

        //------------> Main/first scene setup <---------------
        scene2 = new Scene(layout2,600,550);
        scene2.getStylesheets().add("/style.css");
        window.setResizable(false);
        window.getIcons().add(new Image("microorganism.png"));

    }

    // function to check model exist or not
    private void loadCovid19Model(TextField text) throws IOException{

        File modelSave = new ClassPathResource("trained_covid_model.zip").getFile();
        if (modelSave.exists() == false) {
            text.setText("Model not exist....");
            return;
        } else
            text.setText("Model exist");
    }

    // function to test the uploaded image
    private void testImage(File img, TextField textField) throws IOException{

        int height = 299;
        int width = 299;
        int channel = 1;

        File modelSave = new ClassPathResource("trained_covid_model.zip").getFile();
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);

        NativeImageLoader loader = new NativeImageLoader(height,width,channel);
        INDArray image = loader.asMatrix(img);

//        DataNormalization scaler = new ImagePreProcessingScaler();
//        scaler.transform(image);
        //textField.setText("Image matrix: " + image);

        INDArray  output = model.output(image);
        textField.setText("Label: " + Nd4j.argMax(output,1));
        System.out.println("Prob: " + output.toString());
        //textField.setText("Probabilities: " + output.toString());

    }

    public static void main(String[] args) {
        launch(args);
    }
}
