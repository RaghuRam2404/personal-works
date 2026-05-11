// DAD FIT Website JavaScript

document.addEventListener('DOMContentLoaded', function() {
    
    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    const navbarHeight = navbar.offsetHeight;
    
    // Debounce scroll handler for navbar
    const handleNavbarScroll = debounce(function() {
        const scrollThreshold = 20; // Add a buffer to prevent oscillation
        const navCollapse = document.querySelector('.navbar-collapse');
        const isMenuOpen = navCollapse && navCollapse.classList.contains('show');
        
        if (window.scrollY > scrollThreshold || isMenuOpen) {
            navbar.classList.add('scrolled');
            navbar.classList.remove('navbar-dark');
        } else {
            navbar.classList.remove('scrolled');
            navbar.classList.add('navbar-dark');
        }
    }, 10); // Small delay for smooth transition

    window.addEventListener('scroll', handleNavbarScroll);

    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - navbarHeight;
                
                // Update URL hash
                history.pushState(null, null, targetId);
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                const navbarCollapse = document.querySelector('.navbar-collapse');
                if (navbarCollapse.classList.contains('show')) {
                    const navbarToggler = document.querySelector('.navbar-toggler');
                    navbarToggler.click();
                }
            }
        });
    });

    // Active navigation highlighting
    const sections = document.querySelectorAll('section[id]');
    
    window.addEventListener('scroll', function() {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - navbarHeight - 50;
            const sectionHeight = section.offsetHeight;
            
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });

    // Button click handlers
    const surveyBtn = document.querySelector('.btn-survey');
    const storyBtn = document.querySelector('.btn-story');
    
    // Remove preventDefault for external links
    // surveyBtn and storyBtn now link to external URLs (WhatsApp, YouTube)
    // No need to prevent default behavior
    
    if (storyBtn) {
        // Story button now links to external YouTube URL
        // No need to prevent default behavior
    }

    // Responsive image handling
    function handleResponsiveImages() {
        const heroImage = document.querySelector('.hero-main-image');
        if (heroImage) {
            // Ensure images are properly loaded and displayed
            heroImage.addEventListener('load', function() {
                this.style.opacity = '1';
            });
            
            // Add error handling for images
            heroImage.addEventListener('error', function() {
                this.style.display = 'none';
                console.warn('Hero image failed to load');
            });
        }
    }

    // Initialize responsive image handling
    handleResponsiveImages();

    // Parallax effect for hero background (optional enhancement)
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const heroBackground = document.querySelector('.hero-background');
        
        if (heroBackground && window.innerWidth > 768) {
            const speed = scrolled * 0.5;
            heroBackground.style.transform = `translateY(${speed}px)`;
        }
    });

    // Mobile menu auto-close on resize
    window.addEventListener('resize', function() {
        const navbarCollapse = document.querySelector('.navbar-collapse');
        if (window.innerWidth > 991 && navbarCollapse.classList.contains('show')) {
            const navbarToggler = document.querySelector('.navbar-toggler');
            navbarToggler.click();
        }
    });

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animateElements = document.querySelectorAll('.hero-content > *');
    animateElements.forEach(el => {
        observer.observe(el);
    });

    // Performance optimization: Debounce scroll events
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Apply debouncing to scroll-heavy functions
    const debouncedScroll = debounce(function() {
        // Any heavy scroll operations can go here
    }, 16); // ~60fps

    window.addEventListener('scroll', debouncedScroll);

    // Accessibility enhancements
    function enhanceAccessibility() {
        // Add skip link for keyboard navigation
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-link visually-hidden-focusable';
        skipLink.textContent = 'Skip to main content';
        document.body.insertBefore(skipLink, document.body.firstChild);

        // Enhanced focus management for mobile menu
        const navbarToggler = document.querySelector('.navbar-toggler');
        const navbarCollapse = document.querySelector('.navbar-collapse');
        
        if (navbarToggler && navbarCollapse) {
            // Listen for Bootstrap's collapse events instead of click
            navbarCollapse.addEventListener('show.bs.collapse', function() {
                // When menu is about to open
                navbar.classList.add('scrolled');
                navbar.classList.remove('navbar-dark');
            });

            navbarCollapse.addEventListener('hidden.bs.collapse', function() {
                // When menu is fully closed
                const scrollThreshold = 20;
                const isScrolled = window.scrollY > scrollThreshold;
                
                if (!isScrolled) {
                    navbar.classList.remove('scrolled');
                    navbar.classList.add('navbar-dark');
                }
            });

            // Focus management
            /*navbarCollapse.addEventListener('shown.bs.collapse', function() {
                const firstNavLink = navbarCollapse.querySelector('.nav-link');
                if (firstNavLink) firstNavLink.focus();
            });*/
        }
    }

    // Initialize accessibility enhancements
    enhanceAccessibility();

    // Auto-populate YouTube video thumbnails
    function populateVideoThumbnails() {
        const videoLinks = document.querySelectorAll('.video-link');
        
        videoLinks.forEach(link => {
            const href = link.getAttribute('href');
            const img = link.querySelector('.video-preview');
            
            if (href && img) {
                // Extract video ID from YouTube URL
                let videoId = null;
                
                // Handle youtube.com/watch?v=VIDEO_ID format
                if (href.includes('youtube.com/watch?v=')) {
                    const urlParams = new URLSearchParams(href.split('?')[1]);
                    videoId = urlParams.get('v');
                }
                // Handle youtu.be/VIDEO_ID format
                else if (href.includes('youtu.be/')) {
                    videoId = href.split('youtu.be/')[1].split('?')[0];
                }
                // Handle youtube.com/shorts/VIDEO_ID format
                else if (href.includes('youtube.com/shorts/')) {
                    videoId = href.split('/shorts/')[1].split('?')[0];
                }
                
                // Set thumbnail if video ID found
                if (videoId) {
                    img.src = `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;
                    img.onerror = function() {
                        // Fallback to default quality if mqdefault fails
                        this.src = `https://img.youtube.com/vi/${videoId}/default.jpg`;
                    };
                }
            }
        });
    }

    // Initialize video thumbnails
    populateVideoThumbnails();

    // Resources page navigation smooth scroll
    const resourcesNavLinks = document.querySelectorAll('.resources-nav-link');
    
    resourcesNavLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            const resourcesNav = document.getElementById('resourcesNav');
            
            if (targetSection) {
                const navbarHeight = navbar ? navbar.offsetHeight : 80;
                const resourcesNavHeight = resourcesNav ? resourcesNav.offsetHeight : 60;
                const offsetTop = targetSection.offsetTop - navbarHeight - resourcesNavHeight - 20;
                
                // Update URL hash
                history.pushState(null, null, targetId);
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Highlight active section on scroll
    if (resourcesNavLinks.length > 0) {
        const sections = document.querySelectorAll('section[id]');
        const resourcesNav = document.getElementById('resourcesNav');
        
        window.addEventListener('scroll', function() {
            const navbarHeight = navbar ? navbar.offsetHeight : 80;
            const resourcesNavHeight = resourcesNav ? resourcesNav.offsetHeight : 60;
            const scrollPosition = window.scrollY + navbarHeight + resourcesNavHeight + 100;
            
            let currentSection = '';
            
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;
                
                if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                    currentSection = section.getAttribute('id');
                }
            });
            
            resourcesNavLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + currentSection) {
                    link.classList.add('active');
                }
            });
        });
    }

    console.log('DAD FIT website initialized successfully!');
});