package izzah.mp.covid.ui;

import javafx.scene.image.Image;

// Object covidXray for xray image
public class covidXray {
    private Image xray;

    // Getter and setter that will be used the get and set the uploaded image by the user
    // Getter constructor
    public Image getXray() {
        return xray;
    }

    // Setter Constructor
    public void setXray(Image image) {
        this.xray = image;
    }
}
