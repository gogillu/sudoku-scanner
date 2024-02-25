import cv2

class VideoProjector:
    @staticmethod
    def project_video(frames):
        for frame in frames:
            cv2.imshow('Processed Video Feed', frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):  # Press 'q' to quit the projection
                break
        cv2.destroyAllWindows()
