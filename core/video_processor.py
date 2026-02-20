import sys
import cv2
from pathlib import Path


def process_video(source_face, face_analyser, args):
    """Process a video file"""
    import cv2
    from core.engine.face_swapper import swap_face
    import time

    print(f"Processing video: {args.target}")

    # Open video
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {args.target}")
        sys.exit(1)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # Prepare output
    if args.output:
        output_path = args.output
    else:
        target_path = Path(args.target)
        output_path = f"{target_path.stem}_swapped{target_path.suffix}"

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    print("Processing frames...")
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and swap faces
            faces = face_analyser.get(frame)
            for face in faces:
                frame = swap_face(source_face, face, frame)

            # Write frame
            out.write(frame)

            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(
                    f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_actual:.1f} FPS"
                )

    finally:
        cap.release()
        out.release()

        elapsed = time.time() - start_time
        print(f"\n✓ Video processing complete!")
        print(f"  Processed {frame_count} frames in {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count / elapsed:.1f}")
        print(f"  Output saved: {output_path}")
