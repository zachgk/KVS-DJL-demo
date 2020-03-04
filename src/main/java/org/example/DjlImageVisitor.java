package org.example;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
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

public class DjlImageVisitor extends H264FrameDecoder {

    Predictor<BufferedImage, Classifications> predictor;

    public DjlImageVisitor() throws IOException, ModelNotFoundException, MalformedModelException {
        predictor = MxModelZoo.RESNET.loadModel().newPredictor();
    }

    @Override
    public void process(Frame frame, MkvTrackMetadata trackMetadata,
        Optional<FragmentMetadata> fragmentMetadata) throws FrameProcessException {

        BufferedImage bufferedImage = decodeH264Frame(frame, trackMetadata);

        try {
            Classifications prediction = predictor.predict(bufferedImage);
            System.out.println(prediction.topK(5).toString());
        } catch (TranslateException e) {
            throw new FrameProcessException("Failed to predict", e);
        }
    }
}
