import cv2
from pyzbar.pyzbar import decode

# Function to decode QR codes in a video stream
def read_qr_code():
    # Open the video capture device (webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Decode QR codes
        decoded_objects = decode(frame)

        # Print decoded information
        for obj in decoded_objects:
            print('Type:', obj.type)
            print('Data:', obj.data)
            print()
            # Draw bounding box around the QR code
            (x, y, w, h) = obj.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('QR Code Scanner', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Example usage
read_qr_code()
