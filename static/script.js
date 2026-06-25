let timerInterval;

function startCountdown() {
    let timeLeft = 5;
    const timerDiv = document.getElementById('countdown-timer');
    const timerSpan = document.getElementById('timer-sec');
    
    timerDiv.style.display = 'block'; // Timer dikhao
    timerSpan.innerText = timeLeft;

    timerInterval = setInterval(() => {
        timeLeft -= 1;
        timerSpan.innerText = timeLeft;
        
        if (timeLeft <= 0) {
            clearInterval(timerInterval);
            timerDiv.style.display = 'none'; // Timer chupao
            alert("5 Seconds Over! Please stop the recording now for best results.");
        }
    }, 1000);
}

// Event listener jo "Record" button ko click hote hi pakar le
document.addEventListener('click', (e) => {
    // Gradio ke record button ka aria-label ya text dhundna
    const isRecordBtn = e.target.ariaLabel === "Start recording" || 
                        (e.target.innerText && e.target.innerText.includes("Record"));
    
    if (isRecordBtn) {
        startCountdown();
    }
});