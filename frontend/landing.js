/**
 * TuneKit Landing Page
 */

// Features Carousel
(function initCarousel() {
    function setupCarousel() {
        const track = document.querySelector('.carousel-track');
        const cards = document.querySelectorAll('.feature-card');
        const prevBtn = document.querySelector('.carousel-btn-prev');
        const nextBtn = document.querySelector('.carousel-btn-next');

        if (!track || !cards.length) return;

        let currentIndex = 0;

        function getCardWidth() {
            return window.innerWidth < 768 ? 280 : 340;
        }

        function getGap() {
            return window.innerWidth < 768 ? 20 : 30;
        }

        function updateCarousel() {
            const cardWidth = getCardWidth();
            const gap = getGap();

            // Calculate offset: center the current card
            // Track is at left: 50%, so we need to offset by half card width plus the slide offset
            const totalCardWidth = cardWidth + gap;
            const offset = -(cardWidth / 2) - (currentIndex * totalCardWidth);

            track.style.transform = `translateX(${offset}px)`;

            cards.forEach((card, index) => {
                card.classList.toggle('active', index === currentIndex);
            });

            prevBtn.disabled = currentIndex === 0;
            nextBtn.disabled = currentIndex === cards.length - 1;
        }

        prevBtn.addEventListener('click', () => {
            if (currentIndex > 0) {
                currentIndex--;
                updateCarousel();
            }
        });

        nextBtn.addEventListener('click', () => {
            if (currentIndex < cards.length - 1) {
                currentIndex++;
                updateCarousel();
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            const featuresSection = document.querySelector('.features');
            const rect = featuresSection.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight && rect.bottom > 0;

            if (isVisible) {
                if (e.key === 'ArrowLeft' && currentIndex > 0) {
                    currentIndex--;
                    updateCarousel();
                } else if (e.key === 'ArrowRight' && currentIndex < cards.length - 1) {
                    currentIndex++;
                    updateCarousel();
                }
            }
        });

        // Touch/swipe support
        let touchStartX = 0;

        track.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });

        track.addEventListener('touchend', (e) => {
            const touchEndX = e.changedTouches[0].screenX;
            const diff = touchStartX - touchEndX;
            const swipeThreshold = 50;

            if (Math.abs(diff) > swipeThreshold) {
                if (diff > 0 && currentIndex < cards.length - 1) {
                    currentIndex++;
                    updateCarousel();
                } else if (diff < 0 && currentIndex > 0) {
                    currentIndex--;
                    updateCarousel();
                }
            }
        }, { passive: true });

        // Handle window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(updateCarousel, 100);
        });

        // Initialize
        updateCarousel();
    }

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupCarousel);
    } else {
        setupCarousel();
    }
})();
