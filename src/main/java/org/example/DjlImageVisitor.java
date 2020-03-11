package org.example;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import com.amazonaws.kinesisvideo.parser.mkv.Frame;
import com.amazonaws.kinesisvideo.parser.mkv.FrameProcessException;
import com.amazonaws.kinesisvideo.parser.utilities.FragmentMetadata;
import com.amazonaws.kinesisvideo.parser.utilities.H264FrameDecoder;
import com.amazonaws.kinesisvideo.parser.utilities.MkvTrackMetadata;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Optional;
import java.util.stream.Collectors;

public class DjlImageVisitor extends H264FrameDecoder {

    Predictor<BufferedImage, DetectedObjects> predictor;

    public DjlImageVisitor() throws IOException, ModelNotFoundException, MalformedModelException {
        predictor = MxModelZoo.SSD.loadModel().newPredictor();
    }

    @Override
    public void process(Frame frame, MkvTrackMetadata trackMetadata,
        Optional<FragmentMetadata> fragmentMetadata) throws FrameProcessException {

        BufferedImage bufferedImage = decodeH264Frame(frame, trackMetadata);

        try {
            DetectedObjects prediction = predictor.predict(bufferedImage);
            String classStr = prediction.items().stream().map(Classification::getClassName).collect(
                Collectors.joining());
            System.out.println("Found objects: " + classStr);
            boolean hasPerson = prediction.items().stream().anyMatch(c -> "person".equals(c.getClassName()) && c.getProbability() > 0.5);
        } catch (TranslateException e) {
            throw new FrameProcessException("Failed to predict", e);
        }
    }
}
