package izzah.mp.covid.model;

import org.imgscalr.Scalr;
import org.nd4j.common.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Augmentation {
    public static void main(String [] args) throws IOException{
        BufferedImage image = null;
        File file = null;

        // Directory of images and the path to save the augmentation image in resources folder
        final File parentDir = new ClassPathResource("/Covid-19 Radiography Dataset/Viral Pneumonia").getFile();
        final File dir = new File("D:\\covidApp\\src\\main\\resources\\train\\Viral Pneumonia");
        File opr,opr2,opf,opc;
        int success = 0;


        try {
            for (final File imgFile : parentDir.listFiles()) {
                image = ImageIO.read(imgFile);
                success +=1;

                // Rotae 90 degree
//                Scalr.Rotation rotation = Scalr.Rotation.CW_90;
//                BufferedImage rotated = Scalr.rotate(image, rotation);
//                String newFile = imgFile.getName() + "_2" + rotation.name() + ".png";

                // Rotate 180 degree
                Scalr.Rotation rotation180 = Scalr.Rotation.CW_180;
                BufferedImage rotated180 = Scalr.rotate(image, rotation180);
                String newFile2 = imgFile.getName() + "_" + rotation180.name() + ".png";

                // Horizontal Flip
//                Scalr.Rotation flipHorz = Scalr.Rotation.FLIP_HORZ;
//                BufferedImage flip = Scalr.rotate(image,flipHorz);
//                String nF = imgFile.getName() + "_" + flipHorz.name() + ".png";

                // Cropping process
                Random rnd = new Random();
                int width = image.getWidth();
                int x = rnd.nextInt(width/2);
                int w = (int) ((0.7 + rnd.nextDouble()/2) * width/2);

                int height = image.getHeight();
                int y = rnd.nextInt(height/2);
                int h = (int) ((0.7 + rnd.nextDouble()/2) * height/2);

                if (x+w > width){
                    w = width - x;
                }
                if (y+h > height){
                    h = height - y;
                }

//                BufferedImage crop = Scalr.crop(image,x,y,w,h);
//                String nC = imgFile.getName() + "_crop" + ".png";

//                opr = new File(dir, newFile);
//                ImageIO.write(rotated, "png", opr);

                opr2 = new File(dir, newFile2);
                ImageIO.write(rotated180, "png", opr2);

//                opf = new File(dir, nF);
//                ImageIO.write(flip, "png", opf);

//                opc = new File(dir, nC);
//                ImageIO.write(crop,"png", opc);
            }
        } catch (IOException e) {
            System.out.println(e);
        }

        // Display no of images that successfully transform
        System.out.println("Success: " + success);
    }
}
