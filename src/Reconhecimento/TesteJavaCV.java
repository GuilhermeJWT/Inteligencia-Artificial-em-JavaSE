/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Reconhecimento;

import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
/**
 *
 * @author guiro
 */
public class TesteJavaCV {
    
    public static void main (String args[]){
        FaceRecognizer r = createEigenFaceRecognizer();
    }
    
}
