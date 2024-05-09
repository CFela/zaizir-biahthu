/* import React, { useState, useEffect } from 'react';

function Gesture() {
    const [gestureData, setGestureData] = useState(null);

    useEffect(() => {
        async function fetchGestureData() {
            try {
                const response = await fetch('/data');
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setGestureData(data);
            } catch (error) {
                console.error('Error fetching gesture data:', error);
            }
        }

        fetchGestureData();
    }, []);

    return (
        <div>
            {gestureData ? (
                <pre>{JSON.stringify(gestureData, null, 2)}</pre>
            ) : (
                <p>Loading gesture data...</p>
            )}
        </div>
    );
}

export default Gesture;
 */













/* 
import React, { useState, useEffect } from 'react';

function Gesture() {
    const [gestureData, setGestureData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function fetchGestureData() {
            try {
                const response = await fetch('http://localhost:5000/data');
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setGestureData(data);
            } catch (error) {
                console.error('Error fetching gesture data:', error);
                setError(error.message);
            }
        }

        fetchGestureData();
    }, []);

    return (
        <div>
            {error ? (
                <p>Error: {error}</p>
            ) : (
                <div>
                    {gestureData ? (
                        <pre>{JSON.stringify(gestureData, null, 2)}</pre>
                    ) : (
                        <p>Loading gesture data...</p>
                    )}
                </div>
            )}
        </div>
    );
}

export default Gesture;
 */













/* 
import React, { useEffect, useRef, useState } from "react";

const Gesture = () => {
  const canvasRef = useRef();
  const imageRef = useRef();
  const videoRef = useRef();

  const [result, setResult] = useState("");

  useEffect(() => {
    async function getCameraStream() {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: true,
      });
  
      if (videoRef.current) {      
        videoRef.current.srcObject = stream;
      }
    };
  
    getCameraStream();
  }, []);
  
  useEffect(() => {
    const interval = setInterval(async () => {
      captureImageFromCamera();

      if (imageRef.current) {
        const formData = new FormData();
        formData.append('image', imageRef.current);

        const response = await fetch('/classify', {
          method: "POST",
          body: formData,
        });

        if (response.status === 200) {
          const text = await response.text();
          setResult(text);
        } else {
          setResult("Error from API.");
        }
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const playCameraStream = () => {
    if (videoRef.current) {
      videoRef.current.play();
    }
  };

  const captureImageFromCamera = () => {
    const context = canvasRef.current.getContext('2d');
    const { videoWidth, videoHeight } = videoRef.current;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);

    canvasRef.current.toBlob((blob) => {
      imageRef.current = blob;
    })
  };

  return (
    <>
      <header>
        <h1>Image Gesture</h1>
      </header>
      <main>
        <video ref={videoRef} onCanPlay={() => playCameraStream()} id="video" />
        <canvas ref={canvasRef} hidden></canvas>
        <p>Currently seeing: {result}</p>
      </main>
    </>
  )
};

export default Gesture;
 */
















/* import React, { useState, useEffect } from 'react';

const Gesture = () => {
  const [videoStream, setVideoStream] = useState('');

  useEffect(() => {
    const fetchVideoStream = async () => {
      try {
        const response = await fetch('/video_feed', {
          headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            Pragma: 'no-cache',
            Expires: '0',
          },
        });
        if (!response.ok) {
          throw new Error('Failed to fetch video stream');
        }
        // Assuming the video stream URL is returned as text/plain content
        const videoUrl = await response.text();
        setVideoStream(videoUrl);
      } catch (error) {
        console.error('Error fetching video stream:', error);
      }
    };

    fetchVideoStream();

    return () => {
      // Clean up by revoking the object URL
      if (videoStream) {
        URL.revokeObjectURL(videoStream);
      }
    };
  }, []); // Only fetch video stream once on component mount

  return (
    <div>
      {videoStream && (
        <video autoPlay controls>
          <source src={videoStream} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      )}
    </div>
  );
};

export default Gesture;
 */















import React, { useState, useEffect } from 'react';

const Gesture = () => {
  const [stream, setStream] = useState(null);

  useEffect(() => {
    const fetchStream = async () => {
      try {
        const response = await fetch('http://localhost:8000/video_feed');
        const stream = await response.blob();
        setStream(URL.createObjectURL(stream));
      } catch (error) {
        console.error('Error fetching video stream:', error);
      }
    };

    fetchStream();

    // Clean up
    return () => {
      if (stream) {
        URL.revokeObjectURL(stream);
      }
    };
  }, []);

  return (
    <div>
      {stream && (
        <video controls autoPlay>
          <source src={stream} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      )}
    </div>
  );
};

export default Gesture;

