const playBtn = document.querySelector('#play');
const playIcon = document.querySelector('#play-icon');
const likeBtn = document.querySelector('#like');
const dislikeBtn = document.querySelector('#dislike');
const titleEl = document.querySelector('#title');
const artistEl = document.querySelector('#artist');
const coverEl = document.querySelector('#cover');
const coverElBG = document.querySelector('#coverBG');
const progressBar = document.querySelector('#progress-bar');
const currentTimeEl = document.querySelector('#current-time');
const durationEl = document.querySelector('#duration');

// Audio and Constants
let audio = new Audio();
const DEFAULT_COVER = 'assets/Vinyl_Red.png';
const DEFAULT_BG_COVER = 'assets/blackSquare.png';

// Utility Function to Check Default Image
function isDefaultImage(url) {
    return !url || url.includes("/img/default/album.jpg");
}

// Format Time (mm:ss)
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
}

// Fetch Recommendation from Backend
async function fetchRecommendation(action = 'next') {
    try {
        console.log(`Fetching recommendation from http://localhost:8000/recommend (action: ${action})...`);
        const res = await fetch("http://localhost:8000/recommend", {
            signal: AbortSignal.timeout(10000)
        });
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const song = await res.json();
        console.log("Received song data:", song);

        // Update UI
        titleEl.textContent = song.track_title || "Unknown Title";
        artistEl.textContent = song.artist_name || "Unknown Artist";
       
        coverEl.src = isDefaultImage(song.image_url) ? DEFAULT_COVER : song.image_url;
        coverElBG.src = isDefaultImage(song.image_url) ? DEFAULT_BG_COVER : song.image_url;
        // Set audio
        audio.src = song.audio_url || "";
        await audio.load();

        // Reset progress bar
        progressBar.value = 0;
        currentTimeEl.textContent = '0:00';
        durationEl.textContent = '0:00';
        playIcon.src = 'assets/Play.png';

        // Update duration when metadata is loaded
        audio.onloadedmetadata = () => {
            durationEl.textContent = formatTime(audio.duration);
            progressBar.max = audio.duration;
        };

        // Update progress bar during playback
        audio.ontimeupdate = () => {
            progressBar.value = audio.currentTime;
            currentTimeEl.textContent = formatTime(audio.currentTime);
        };

        // Seek audio when progress bar is changed
        progressBar.oninput = () => {
            audio.currentTime = progressBar.value;
        };
    } catch (err) {
        console.error("Failed to fetch recommendation:", err);
        titleEl.textContent = "No more recommendations.";
        artistEl.textContent = "";
        coverEl.src = DEFAULT_COVER;
        coverElBG.src = DEFAULT_BG_COVER;
        audio.src = "";
        progressBar.value = 0;
        currentTimeEl.textContent = '0:00';
        durationEl.textContent = '0:00';
    }
}

// Play/Pause Functionality
playBtn.onclick = () => {
    if (!audio.src) {
        console.warn("No audio source available.");
        return;
    }

    if (audio.paused) {
        audio.play()
            .then(() => {
                playIcon.src = 'assets/Pause.png';
                playIcon.alt = 'Pause';
            })
            .catch(err => {
                console.error("Playback failed:", err);
            });
    } else {
        audio.pause();
        playIcon.src = 'assets/Play.png';
        playIcon.alt = 'Play';
    }
};


// Like Action
likeBtn.onclick = () => sendAction('like');

// Share Action (Placeholder)
dislikeBtn.onclick = () => sendAction('dislike');

// Send Action to Backend
async function sendAction(action) {
    try {
        console.log(`Sending ${action} action to http://localhost:8000/action`);
        const response = await fetch('http://localhost:8000/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        await fetchRecommendation(action); // Refresh recommendation after action
    } catch (err) {
        console.error(`Failed to send ${action} action:`, err);
    }
}

// Initial Load
fetchRecommendation();