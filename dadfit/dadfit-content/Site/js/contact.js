// Contact Page Functionality

document.addEventListener('DOMContentLoaded', function() {
    
    // Range slider value updates
    const motivationSlider = document.getElementById('motivationLevel');
    const motivationValue = document.getElementById('motivationLevelValue');
    
    if (motivationSlider && motivationValue) {
        motivationSlider.addEventListener('input', function() {
            motivationValue.textContent = this.value;
        });
    }
    
    // Character counter for textarea
    const messageTextarea = document.getElementById('message');
    const charCount = document.getElementById('charCount');
    
    if (messageTextarea && charCount) {
        messageTextarea.addEventListener('input', function() {
            const count = this.value.length;
            charCount.textContent = count;
            
            // Change color when approaching limit
            if (count > 900) {
                charCount.style.color = '#dc3545';
            } else if (count > 800) {
                charCount.style.color = '#ffc107';
            } else {
                charCount.style.color = 'var(--primary-dark)';
            }
        });
        
        // Enforce max length
        messageTextarea.addEventListener('input', function() {
            if (this.value.length > 1000) {
                this.value = this.value.substring(0, 1000);
                charCount.textContent = 1000;
            }
        });
    }
    
    // Zoho CRM Email Validation Function
    function validateEmail() {
        const emailFld = document.querySelectorAll('[ftype=email]');
        for (let i = 0; i < emailFld.length; i++) {
            const emailVal = emailFld[i].value;
            if ((emailVal.replace(/^\s+|\s+$/g, '')).length != 0) {
                const atpos = emailVal.indexOf('@');
                const dotpos = emailVal.lastIndexOf('.');
                if (atpos < 1 || dotpos < atpos + 2 || dotpos + 2 >= emailVal.length) {
                    alert('Please enter a valid email address.');
                    emailFld[i].focus();
                    return false;
                }
            }
        }
        return true;
    }
    
    // Zoho CRM Mandatory Field Check Function
    function checkMandatory() {
        const mndFileds = ['Last Name', 'Email', 'Mobile', 'LEADCF51', 'LEADCF52', 'Company'];
        const fldLangVal = ['Last Name', 'Email', 'Phone Number', 'Number of Kids', 'Age', 'Company/Profession'];
        
        for (let i = 0; i < mndFileds.length; i++) {
            const fieldObj = document.forms['WebToLeads1166383000000602627'][mndFileds[i]];
            if (fieldObj) {
                if (((fieldObj.value).replace(/^\s+|\s+$/g, '')).length == 0) {
                    if (fieldObj.type == 'file') {
                        alert('Please select a file to upload.');
                        fieldObj.focus();
                        return false;
                    }
                    alert(fldLangVal[i] + ' cannot be empty.');
                    fieldObj.focus();
                    return false;
                } else if (fieldObj.nodeName == 'SELECT') {
                    if (fieldObj.options[fieldObj.selectedIndex].value == '-None-') {
                        alert(fldLangVal[i] + ' cannot be none.');
                        fieldObj.focus();
                        return false;
                    }
                } else if (fieldObj.type == 'checkbox') {
                    if (fieldObj.checked == false) {
                        alert('Please accept ' + fldLangVal[i]);
                        fieldObj.focus();
                        return false;
                    }
                }
            }
        }
        
        if (!validateEmail()) {
            return false;
        }
        
        // Handle smarturl service parameter if present
        const urlparams = new URLSearchParams(window.location.search);
        if (urlparams.has('service') && (urlparams.get('service') === 'smarturl')) {
            const webform = document.getElementById('contactForm');
            const service = urlparams.get('service');
            const smarturlfield = document.createElement('input');
            smarturlfield.setAttribute('type', 'hidden');
            smarturlfield.setAttribute('value', service);
            smarturlfield.setAttribute('name', 'service');
            webform.appendChild(smarturlfield);
        }
        
        return true;
    }
    
    // Form validation and submission
    const contactForm = document.getElementById('contactForm');
    const btnSubmit = document.querySelector('.btn-submit');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.querySelector('.btn-loader');
    
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Remove previous validation states
            const inputs = contactForm.querySelectorAll('.form-control, .form-select');
            inputs.forEach(input => {
                input.classList.remove('is-invalid', 'is-valid');
            });
            
            // Check form validity
            let isValid = true;
            
            // Validate required text inputs
            const requiredInputs = contactForm.querySelectorAll('input[required], select[required], textarea[required]');
            requiredInputs.forEach(input => {
                if (!input.value.trim()) {
                    input.classList.add('is-invalid');
                    isValid = false;
                }
            });
            
            // Validate email
            const emailInput = document.getElementById('email');
            if (emailInput && emailInput.value) {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(emailInput.value)) {
                    emailInput.classList.add('is-invalid');
                    emailInput.classList.remove('is-valid');
                    isValid = false;
                }
            }
            
            // Validate phone
            const phoneInput = document.getElementById('phone');
            if (phoneInput && phoneInput.value) {
                const phoneRegex = /^[0-9]{10}$/;
                if (!phoneRegex.test(phoneInput.value)) {
                    phoneInput.classList.add('is-invalid');
                    phoneInput.classList.remove('is-valid');
                    isValid = false;
                }
            }
            
            // Validate How Did You Hear About Us
            const howHearUsSelect = document.getElementById('howHearUs');
            if (howHearUsSelect && howHearUsSelect.hasAttribute('required')) {
                if (!howHearUsSelect.value || howHearUsSelect.value === '-None-') {
                    howHearUsSelect.classList.add('is-invalid');
                    howHearUsSelect.classList.remove('is-valid');
                    isValid = false;
                } else {
                    howHearUsSelect.classList.add('is-valid');
                    howHearUsSelect.classList.remove('is-invalid');
                }
            }
            
            if (!isValid) {
                // Scroll to first invalid field
                const firstInvalid = contactForm.querySelector('.is-invalid');
                if (firstInvalid) {
                    firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    firstInvalid.focus();
                }
                return;
            }
            
            // Run Zoho CRM validation
            if (!checkMandatory()) {
                return;
            }
            
            // Show loading state
            btnSubmit.disabled = true;
            btnText.classList.add('d-none');
            btnLoader.classList.remove('d-none');
            
            // Set character encoding
            document.charset = 'UTF-8';
            
            // Submit form to Zoho CRM (will redirect to contact_success.html)
            contactForm.submit();
        });
    }
    
    // Real-time validation on blur/focusout
    const formInputs = document.querySelectorAll('#contactForm input, #contactForm select, #contactForm textarea');
    formInputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateField(this);
        });
        
        // Re-validate on input for email, phone, age, and kids fields
        input.addEventListener('input', function() {
            if (this.classList.contains('is-invalid')) {
                this.classList.remove('is-invalid');
            }
            // Re-validate specific fields on input if they were previously validated
            if ((this.getAttribute('ftype') === 'email' || this.id === 'phone' || this.id === 'age' || this.id === 'kids') && 
                (this.classList.contains('is-valid') || this.classList.contains('is-invalid'))) {
                validateField(this);
            }
        });
    });
    
    // Field validation function
    function validateField(field) {
        const value = field.value.trim();
        
        // Skip validation if field is empty and not required
        if (!value && !field.hasAttribute('required')) {
            field.classList.remove('is-invalid', 'is-valid');
            return;
        }
        
        let isValid = true;
        
        // Check if required field is empty
        if (field.hasAttribute('required') && !value) {
            isValid = false;
        }
        // Validate age (20-60)
        else if (field.id === 'age') {
            const age = parseInt(value);
            isValid = !isNaN(age) && age >= 20 && age <= 60;
        }
        // Validate kids (1-10)
        else if (field.id === 'kids') {
            const kids = parseInt(value);
            isValid = !isNaN(kids) && kids >= 1 && kids <= 10;
        }
        // Validate email
        else if (field.getAttribute('ftype') === 'email') {
            if (value) {
                const emailRegex = /^[^\s@]+(\+[^\s@]+)?@[^\s@]+\.[^\s@]+$/;
                isValid = emailRegex.test(value);
            } else if (field.hasAttribute('required')) {
                isValid = false;
            } else {
                // Not required and empty - don't show any validation
                field.classList.remove('is-invalid', 'is-valid');
                return;
            }
        }
        // Validate phone (10 digits)
        else if (field.id === 'phone') {
            if (value) {
                const phoneRegex = /^[0-9]{10}$/;
                isValid = phoneRegex.test(value);
            } else if (field.hasAttribute('required')) {
                isValid = false;
            } else {
                // Not required and empty - don't show any validation
                field.classList.remove('is-invalid', 'is-valid');
                return;
            }
        }
        // Validate select fields
        else if (field.nodeName === 'SELECT' && field.hasAttribute('required')) {
            isValid = value && value !== '-None-';
        }
        
        // Apply validation classes - only show invalid state, no green valid state
        if (isValid) {
            field.classList.remove('is-invalid', 'is-valid');
        } else {
            field.classList.add('is-invalid');
            field.classList.remove('is-valid');
        }
    }
});
