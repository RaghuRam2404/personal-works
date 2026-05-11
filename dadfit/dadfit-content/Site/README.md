# DAD FIT Website

A responsive fitness website designed for Indian Dads, built with HTML5, CSS3, Bootstrap 5, and vanilla JavaScript.

## 🚀 Features

- **Fully Responsive Design**: Optimized for mobile, tablet, and desktop devices
- **Modern UI/UX**: Clean, professional design with smooth animations
- **Bootstrap 5**: Latest Bootstrap framework for reliable responsive behavior
- **Accessibility**: WCAG compliant with proper ARIA labels and keyboard navigation
- **Performance Optimized**: Lightweight and fast loading
- **Cross-browser Compatible**: Works on all modern browsers

## 📱 Responsive Breakpoints

### Mobile (320px - 767px)
- Single column layout
- Stacked navigation in hamburger menu
- Centered content alignment
- Touch-optimized buttons
- Optimized typography for small screens

### Tablet (768px - 1024px)
- Optimized for both portrait and landscape orientations
- Balanced content layout
- Touch-friendly interface
- Proper spacing for medium screens

### Desktop (1025px+)
- Two-column hero layout
- Hero image visible on right side
- Full navigation menu
- Optimal typography and spacing
- Hover effects and interactions

## 🛠️ Technologies Used

- **HTML5**: Semantic markup with accessibility features
- **CSS3**: Modern styling with flexbox, grid, and animations
- **Bootstrap 5.3.2**: Responsive framework
- **JavaScript (ES6+)**: Modern vanilla JavaScript
- **Google Fonts**: Inter font family
- **Responsive Images**: Optimized for different screen densities

## 📂 Project Structure

```
Site/
├── index.html              # Main HTML file
├── responsive-test.html    # Testing utility page
├── css/
│   └── styles.css         # Custom CSS styles
├── js/
│   └── script.js          # JavaScript functionality
├── images/
│   ├── Dadfit logo.jpg    # Brand logo
│   ├── Rectangle 1.png    # Hero image
│   └── Asset 1 7.svg      # Play icon
└── Site Basic Info.md     # Color scheme and design info
```

## 🎨 Color Scheme

Based on `Site Basic Info.md`:
- **Primary Green**: #34C363
- **Dark Background**: #1E1E1E
- **White**: #FFFFFF
- **Gray**: #9A9A9A
- **Light Gray**: #F4F4F4

## 🚀 Getting Started

### Local Development

1. **Clone or download the project files**

2. **Start a local server**:
   ```bash
   # Using Python 3
   cd /path/to/Site
   python3 -m http.server 8000
   
   # Using Node.js (if you have it installed)
   npx http-server -p 8000
   
   # Using PHP (if you have it installed)
   php -S localhost:8000
   ```

3. **Open in browser**:
   - Main site: `http://localhost:8000`
   - Responsive test: `http://localhost:8000/responsive-test.html`

### Testing Responsive Design

#### Method 1: Browser Developer Tools
1. Open the website in Chrome, Firefox, or Safari
2. Press `F12` or right-click → "Inspect Element"
3. Click the device icon (mobile/tablet icon)
4. Test different device presets:
   - iPhone 12/13/14
   - iPad Air
   - iPad Pro
   - Samsung Galaxy devices
   - Generic mobile/tablet sizes

#### Method 2: Responsive Test Page
1. Navigate to `http://localhost:8000/responsive-test.html`
2. Resize your browser window
3. View real-time viewport information
4. Test different screen sizes

#### Method 3: Manual Resize
1. Open the main site
2. Manually resize browser window
3. Observe layout changes at different breakpoints
4. Test both portrait and landscape orientations

## 📱 Device Testing Checklist

### Mobile Testing (320px - 767px)
- [ ] Navigation collapses to hamburger menu
- [ ] Hero title is readable and properly sized
- [ ] Buttons stack vertically and are touch-friendly
- [ ] Content is centered and properly spaced
- [ ] Images scale appropriately
- [ ] Text remains legible

### Tablet Testing (768px - 1024px)
- [ ] Layout adapts to medium screen sizes
- [ ] Navigation works in both orientations
- [ ] Hero section maintains visual balance
- [ ] Buttons are appropriately sized
- [ ] Content spacing is optimal

### Desktop Testing (1025px+)
- [ ] Two-column layout displays correctly
- [ ] Hero image appears on desktop
- [ ] Navigation is fully visible
- [ ] Hover effects work properly
- [ ] Typography is optimal for large screens

### Cross-Browser Testing
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)
- [ ] Mobile browsers (iOS Safari, Chrome Mobile)

## 🎯 Key Features Implemented

### Navigation
- Fixed top navigation with scroll effects
- Mobile hamburger menu
- Smooth scrolling to sections
- Active section highlighting

### Hero Section
- Responsive background with CSS gradients
- Animated content entrance
- Call-to-action buttons
- Responsive typography

### Accessibility
- Proper ARIA labels
- Keyboard navigation support
- Screen reader friendly
- Skip links for keyboard users
- High contrast ratios

### Performance
- Optimized images
- Minimal JavaScript
- Efficient CSS
- Fast loading times

## 🔧 Customization

### Colors
Modify the CSS custom properties in `styles.css`:
```css
:root {
    --primary-green: #34C363;
    --primary-dark: #1E1E1E;
    /* ... other colors */
}
```

### Breakpoints
Adjust responsive breakpoints in the CSS media queries as needed.

### Content
Edit the HTML content in `index.html` to match your specific requirements.

## 📞 Support

For questions or issues with the responsive design, test the website thoroughly using the methods outlined above. The design has been optimized for:

- **Mobile**: 320px to 767px
- **Tablet**: 768px to 1024px  
- **Desktop**: 1025px and above

All major browsers and devices are supported with fallbacks for older browsers.