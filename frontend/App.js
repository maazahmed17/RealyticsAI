// Global Negotiation State
let currentNegotiationContext = null; // { propertyId, price?, details }
let currentNegotiationSessionId = null; // session id from backend

/**
 * RealyticsAI - Interactive Application Logic
 * Handles UI interactions, animations, and chat functionality
 */

class RealyticsAI {
    constructor() {
        this.currentConversationId = null;
        this.conversations = [];
        this.isTyping = false;
        this.messageHistory = [];
        this.isFirstTimeUser = !localStorage.getItem('returningUser');
        this.currentChart = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadConversationHistory();
        this.initializeWelcomeScreen();
    }

    initializeElements() {
        // Sidebars
        this.leftSidebar = document.getElementById('leftSidebar');
        this.rightSidebar = document.getElementById('rightSidebar');
        this.leftSidebarToggle = document.getElementById('leftSidebarToggle');
        this.rightSidebarToggle = document.getElementById('rightSidebarToggle');
        this.closeRightSidebar = document.getElementById('closeRightSidebar');
        
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.welcomeContainer = document.getElementById('welcomeContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // Negotiation bar elements
        this.negotiationBar = document.getElementById('negotiation-bar');
        this.negotiatePropertyDetails = document.getElementById('negotiate-property-details');
        this.startNegotiationBtn = document.getElementById('start-negotiation-btn');
        this.closeNegotiationBarBtn = document.getElementById('close-negotiation-bar');
        
        // Features menu
        this.featuresBtn = document.getElementById('featuresBtn');
        this.featuresMenu = document.getElementById('featuresMenu');
        
        // Other elements
        this.newChatBtn = document.getElementById('newChatBtn');
        this.conversationHistory = document.getElementById('conversationHistory');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        
        // Quick action cards
        this.quickActionCards = document.querySelectorAll('.quick-action-card');
        this.queryChips = document.querySelectorAll('.query-chip');
    }

    attachEventListeners() {
        // Sidebar toggles
        this.leftSidebarToggle.addEventListener('click', () => this.toggleLeftSidebar());
        this.rightSidebarToggle.addEventListener('click', () => this.toggleRightSidebar());
        this.closeRightSidebar.addEventListener('click', () => this.toggleRightSidebar());
        
        // Input handling
        this.messageInput.addEventListener('input', () => this.handleInputChange());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyPress(e));
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Features menu
        this.featuresBtn.addEventListener('click', () => this.toggleFeaturesMenu());
        document.addEventListener('click', (e) => this.handleDocumentClick(e));
        
        // Feature items
        const featureItems = document.querySelectorAll('.feature-item:not(.disabled)');
        featureItems.forEach(item => {
            item.addEventListener('click', () => this.selectFeature(item));
        });
        
        // Quick actions
        this.quickActionCards.forEach(card => {
            if (!card.classList.contains('disabled')) {
                card.addEventListener('click', () => this.handleQuickAction(card));
            }
        });
        
        // Sample queries
        this.queryChips.forEach(chip => {
            chip.addEventListener('click', () => this.handleQueryChip(chip));
        });
        
        // New chat button
        this.newChatBtn.addEventListener('click', () => this.startNewChat());

        // Start negotiation button
        if (this.startNegotiationBtn) {
            this.startNegotiationBtn.addEventListener('click', () => this.startNegotiationFlow());
        }
        // Close negotiation bar
        if (this.closeNegotiationBarBtn) {
            this.closeNegotiationBarBtn.addEventListener('click', () => {
                if (this.negotiationBar) this.negotiationBar.style.display = 'none';
                currentNegotiationContext = null;
            });
        }
    }

    // Sidebar Management
    toggleLeftSidebar() {
        this.leftSidebar.classList.toggle('collapsed');
    }

    toggleRightSidebar() {
        this.rightSidebar.classList.toggle('hidden');
        // Also toggle chat container margin for side-by-side view
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.classList.toggle('sidebar-open', !this.rightSidebar.classList.contains('hidden'));
        }
    }

    // Features Menu
    toggleFeaturesMenu() {
        this.featuresMenu.classList.toggle('show');
    }

    handleDocumentClick(e) {
        // Close features menu if clicked outside
        if (!this.featuresBtn.contains(e.target) && !this.featuresMenu.contains(e.target)) {
            this.featuresMenu.classList.remove('show');
        }
    }

    selectFeature(item) {
        const feature = item.dataset.feature;
        const prompts = {
            'auto': '', // Auto mode - no prefix, let AI identify intent
            'recommendation': 'Find me properties: ',
            'valuation': 'Estimate the price of a property: ',
            'negotiation': 'Help me negotiate for: '
        };
        
        if (feature === 'auto') {
            // Auto mode - focus input without prefix
            this.messageInput.focus();
            this.messageInput.placeholder = 'Ask anything about properties - AI will understand automatically...';
        } else if (prompts[feature]) {
            this.messageInput.value = prompts[feature];
            this.messageInput.focus();
            this.handleInputChange();
        }
        
        this.featuresMenu.classList.remove('show');
    }

    // Input Handling
    handleInputChange() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText;
        
        // Auto-resize textarea
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    handleKeyPress(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendBtn.disabled) {
                this.sendMessage();
            }
        }
    }

    // Quick Actions & Chips
    handleQuickAction(card) {
        const action = card.dataset.action;
        if (action === 'price') {
            this.messageInput.value = 'I need a price estimate for my property';
            this.messageInput.focus();
            this.handleInputChange();
        }
    }

    handleQueryChip(chip) {
        const query = chip.textContent.replace(/['"]/g, '');
        this.messageInput.value = query;
        this.sendMessage();
    }

    // Message Handling
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Hide welcome container if visible
        if (this.welcomeContainer) {
            this.welcomeContainer.style.display = 'none';
        }
        
        // Add user message
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.handleInputChange();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Simulate API call and response
        await this.getAIResponse(message);
    }

    addMessage(content, sender = 'user', includeActions = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        let actionsHtml = '';
        if (sender === 'assistant' && includeActions) {
            actionsHtml = `
                <div class="message-actions">
                    <button class="action-btn" data-action="copy" title="Copy">
                        <i class="ph ph-copy"></i>
                    </button>
                    <button class="action-btn" data-action="share" title="Share">
                        <i class="ph ph-share-network"></i>
                    </button>
                    <button class="action-btn" data-action="explain" title="Explain">
                        <i class="ph ph-lightbulb"></i>
                    </button>
                    <button class="action-btn" data-action="refine" title="Refine">
                        <i class="ph ph-pencil-simple"></i>
                    </button>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="ph ph-${sender === 'user' ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(content)}</div>
                <div class="message-time">${this.getCurrentTime()}</div>
            </div>
            ${actionsHtml}
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        if (actionsHtml) {
            this.attachActionButtonListeners(messageDiv);
        }
        
        this.messageHistory.push({ content, sender, timestamp: new Date() });
    }

    showTypingIndicator() {
        this.isTyping = true;
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-message';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="ph ph-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const typingMessage = document.querySelector('.typing-message');
        if (typingMessage) {
            typingMessage.remove();
        }
        this.isTyping = false;
    }

    async getAIResponse(message) {
        try {
            // Call the API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator();
            
            // Add AI message with streaming effect
            await this.addStreamingMessage(data.response);
            
            // Show valuation card if it's a valuation response
            if (data.is_price_prediction && data.property_details) {
                const priceMatch = data.response.match(/₹([\d,.]+)\s*(?:lakhs?|crores?)/i);
                let actualPrice = 185; // Default
                if (priceMatch) {
                    const priceText = priceMatch[1].replace(/,/g, '');
                    actualPrice = parseFloat(priceText);
                    if (data.response.toLowerCase().includes('crore')) {
                        actualPrice *= 100;
                    }
                }
                this.showValuationWithData(data.property_details, actualPrice, data.response);
            }
            
            // Show recommendations if available
            if (data.is_recommendation && data.recommendations && data.recommendations.length > 0) {
                this.showRecommendationsView(data.recommendations, data.total_recommendations);
            }

            // NEW: Detect property context for negotiation enablement
            try {
                let detectedPropertyId = null;
                let detailsText = null;
                
                // Check for property_details in response (from price prediction)
                if (data.property_details && (data.property_details.id || data.property_details.property_id)) {
                    detectedPropertyId = data.property_details.id || data.property_details.property_id;
                    const loc = data.property_details.location || 'Selected Property';
                    const sqft = data.property_details.sqft || data.property_details.total_sqft || '';
                    const bhk = data.property_details.bhk ? `${data.property_details.bhk} BHK` : '';
                    
                    // Extract predicted price from response text or actualPrice variable
                    let predictedPrice = null;
                    if (data.is_price_prediction) {
                        const priceMatch = data.response.match(/₹([\d,.]+)\s*(?:lakhs?|l)/i);
                        if (priceMatch) {
                            predictedPrice = parseFloat(priceMatch[1].replace(/,/g, ''));
                        }
                    }
                    
                    detailsText = `${loc}${sqft ? ` · ${sqft} sqft` : ''}${bhk ? ` · ${bhk}` : ''}${predictedPrice ? ` · ₹${predictedPrice.toFixed(1)} Lakhs` : ''}`;
                    
                    // Store context with predicted price as asking price
                    currentNegotiationContext = {
                        propertyId: detectedPropertyId,
                        details: detailsText,
                        askingPrice: predictedPrice,
                        location: loc,
                        bhk: bhk,
                        sqft: sqft
                    };
                    
                    console.log('Negotiation context set from prediction:', currentNegotiationContext);
                }
                // Check for property recommendations
                else if (data.recommendations && data.recommendations.length > 0) {
                    const firstProp = data.recommendations[0];
                    detectedPropertyId = firstProp.id || firstProp.property_id || `prop-${Date.now()}`;
                    const loc = firstProp.location || firstProp.area_type || 'Recommended Property';
                    const sqft = firstProp.sqft || firstProp.total_sqft || '';
                    const bhk = firstProp.bhk ? `${firstProp.bhk} BHK` : '';
                    const price = firstProp.price || firstProp.estimated_price || firstProp.list_price || null;
                    const askingPrice = price ? parseFloat(price) : null;
                    detailsText = `${loc}${sqft ? ` · ${sqft} sqft` : ''}${bhk ? ` · ${bhk}` : ''}${askingPrice ? ` · ₹${askingPrice.toFixed(1)} Lakhs` : ''}`;
                    
                    // Store full context with asking price
                    currentNegotiationContext = {
                        propertyId: detectedPropertyId,
                        details: detailsText,
                        askingPrice: askingPrice,
                        location: loc,
                        bhk: bhk,
                        sqft: sqft
                    };
                    
                    console.log('Negotiation context set:', currentNegotiationContext);
                }
                // Check for property mention in response text
                else if (typeof data.response === 'string' && data.response.toLowerCase().includes('property')) {
                    const idMatch = data.response.match(/(pid-[\w-]+|prop-[\w-]+)/i);
                    detectedPropertyId = idMatch ? idMatch[1] : `prop-${Date.now()}`;
                    detailsText = 'Property mentioned in response';
                }
                
                // Update negotiation bar display
                if (currentNegotiationContext && currentNegotiationContext.propertyId) {
                    if (this.negotiatePropertyDetails) {
                        this.negotiatePropertyDetails.innerText = currentNegotiationContext.details;
                    }
                    if (this.negotiationBar) {
                        this.negotiationBar.style.display = 'flex';
                        console.log('Negotiation bar shown for property:', currentNegotiationContext);
                    }
                }
            } catch (e) {
                console.warn('Negotiation detection failed:', e);
            }
            
        } catch (error) {
            console.error('Error getting AI response:', error);
            await this.delay(1500);
            this.removeTypingIndicator();
            let response = this.generateSampleResponse(message);
            await this.addStreamingMessage(response);
        }
    }

    async addStreamingMessage(content) {
        const isValuation = content.toLowerCase().includes('price') || 
                           content.toLowerCase().includes('estimate') ||
                           content.toLowerCase().includes('valuation');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        let actionsHtml = '';
        if (isValuation) {
            actionsHtml = `
                <div class="message-actions">
                    <button class="action-btn" data-action="copy" title="Copy">
                        <i class="ph ph-copy"></i>
                    </button>
                    <button class="action-btn" data-action="share" title="Share">
                        <i class="ph ph-share-network"></i>
                    </button>
                    <button class="action-btn" data-action="explain" title="Explain (Coming Soon)">
                        <i class="ph ph-lightbulb"></i>
                    </button>
                    <button class="action-btn" data-action="refine" title="Refine">
                        <i class="ph ph-pencil-simple"></i>
                    </button>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="ph ph-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text"><span class="streaming-text"></span></div>
                <div class="message-time">${this.getCurrentTime()}</div>
            </div>
            ${actionsHtml}
        `;
        
        this.chatMessages.appendChild(messageDiv);
        
        const streamingText = messageDiv.querySelector('.streaming-text');
        const words = content.split(' ');
        
        for (let i = 0; i < words.length; i++) {
            streamingText.textContent += (i > 0 ? ' ' : '') + words[i];
            this.scrollToBottom();
            await this.delay(30);
        }
        
        streamingText.classList.remove('streaming-text');
        
        const messageTextDiv = messageDiv.querySelector('.message-text');
        if (messageTextDiv) {
            messageTextDiv.innerHTML = this.formatMessage(content);
        }
        
        if (actionsHtml) {
            this.attachActionButtonListeners(messageDiv);
        }
        
        this.messageHistory.push({ content, sender: 'assistant', timestamp: new Date() });
        
        if (content.toLowerCase().includes('whitefield') || content.toLowerCase().includes('koramangala')) {
            this.showValuationResults();
        }
    }

    async startNegotiationFlow() {
        try {
            if (!currentNegotiationContext || !currentNegotiationContext.propertyId) {
                alert('Please select a property first.');
                return;
            }
            
            // Get asking price from context
            const askingPrice = currentNegotiationContext.askingPrice;
            if (!askingPrice || askingPrice <= 0) {
                alert('Property price information is not available. Please select a property with price details.');
                return;
            }
            
            const input = prompt(`Property asking price is ₹${askingPrice.toFixed(1)} Lakhs.\n\nWhat is your target price in Lakhs?`);
            if (!input) return;
            
            const targetPrice = parseFloat(input);
            if (isNaN(targetPrice) || targetPrice <= 0) {
                alert('Please enter a valid number.');
                return;
            }

            // Show loading in chat
            this.showTypingIndicator();
            
            const resp = await fetch('/api/negotiate/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    property_id: currentNegotiationContext.propertyId,
                    target_price: targetPrice,
                    user_role: 'buyer',
                    initial_message: '',
                    asking_price: askingPrice
                })
            });
            
            this.removeTypingIndicator();
            const data = await resp.json();
            
            if (resp.ok) {
                currentNegotiationSessionId = data.session_id;
                if (this.negotiationBar) {
                    this.negotiationBar.style.display = 'none';
                }
                // Display the negotiation analysis
                await this.addStreamingMessage(data.agent_opening || 'Negotiation analysis completed.');
                
                // Reset negotiation context - this is a one-time analysis
                currentNegotiationSessionId = null;
                currentNegotiationContext = null;
                this.messageInput.placeholder = 'Ask about a property, or select a feature...';
            } else {
                throw new Error(data.detail || 'Failed to start negotiation');
            }
        } catch (e) {
            console.error('Failed to start negotiation:', e);
            this.removeTypingIndicator();
            const msg = (e && e.message) ? e.message : 'Could not analyze negotiation at the moment.';
            this.addMessage(msg, 'assistant');
        }
    }

    generateSampleResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('price') || lowerMessage.includes('estimate')) {
            return "I can help you estimate the property value. Based on the information provided, I'll need a few more details:\n\n" +
                   "• Location (area/neighborhood)\n" +
                   "• Property size (in sq.ft)\n" +
                   "• Number of bedrooms (BHK)\n" +
                   "• Number of bathrooms\n\n" +
                   "Once you provide these details, I can give you an accurate price estimate based on current market data.";
        } else if (lowerMessage.includes('whitefield')) {
            return "Based on the current market data for Whitefield:\n\n" +
                   "• Average price per sq.ft: ₹6,181\n" +
                   "• Market segment: Premium\n" +
                   "• 3 BHK properties typically range from ₹75-120 Lakhs\n\n" +
                   "The exact price would depend on specific factors like exact location, floor level, amenities, and property condition.";
        } else if (lowerMessage.includes('koramangala')) {
            return "Koramangala is one of Bengaluru's premium locations. For a 1500 sq.ft apartment:\n\n" +
                   "• Expected price range: ₹150-180 Lakhs\n" +
                   "• Price per sq.ft: ₹10,000-12,000\n" +
                   "• Market segment: Luxury\n\n" +
                   "This area commands premium prices due to its central location and excellent connectivity.";
        } else {
            return "I'm here to help you with property-related queries in Bengaluru. You can ask me about:\n\n" +
                   "• Property price estimates\n" +
                   "• Market trends for specific areas\n" +
                   "• Comparison between different locations\n" +
                   "• Investment recommendations\n\n" +
                   "Please provide property details or select a feature to get started.";
        }
    }

    initializeWelcomeScreen() {
        const welcomeContainer = document.getElementById('welcomeContainer');
        
        if (this.isFirstTimeUser) {
            welcomeContainer.innerHTML = `
                <div class="welcome-first-time">
                    <div class="welcome-logo">
                        <span class="welcome-logo-r">r</span><span class="welcome-logo-ai">AI</span>
                    </div>
                    <h1 class="welcome-main-title">
                        <span class="welcome-brand">r<span class="brand-ai">AI</span></span> 
                        <span class="welcome-subtitle">RealyticsAI</span>
                    </h1>
                    <p class="welcome-tagline">Property Intelligence Suite</p>
                    <p class="welcome-description">Transforming real estate decisions with AI-powered insights and intelligent market analysis</p>
                    <div class="welcome-instruction">
                        <i class="ph ph-sparkle"></i>
                        <span>Ask a question or click the <strong>+</strong> button to explore features</span>
                    </div>
                </div>
            `;
            localStorage.setItem('returningUser', 'true');
        } else {
            welcomeContainer.innerHTML = `
                <div class="welcome-simple">
                    <p class="welcome-simple-text">How can I help you today?</p>
                </div>
            `;
        }
    }
    
    showValuationWithData(propertyData, actualPrice = 185, responseText = '') {
        // Use actual data from API if available
        const features = propertyData || {
            location: 'Whitefield',
            sqft: 1500,
            bhk: 3,
            bath: 2,
            balcony: 2
        };
        
        // Calculate local average (slightly lower than prediction)
        const localAverage = Math.round(actualPrice * 0.85);
        const priceRange = [Math.round(actualPrice * 0.9), Math.round(actualPrice * 1.1)];
        
        const valuationData = {
            price: actualPrice,
            priceRange: priceRange,
            localAverage: localAverage,
            prediction: actualPrice,
            features: {
                'Location': features.location || 'Not specified',
                'Size': `${features.sqft || 1500} sq.ft`,
                'Bedrooms': `${features.bhk || 3} BHK`,
                'Bathrooms': features.bath || 2,
                'Balconies': features.balcony || 1
            },
            responseText: responseText
        };
        
        this.displayValuationCard(valuationData);
    }
    
    showValuationResults() {
        // Default sample data version
        const valuationData = {
            price: 185,
            priceRange: [175, 195],
            localAverage: 165,
            prediction: 185,
            features: {
                'Location': 'Whitefield',
                'Size': '1500 sq.ft',
                'Bedrooms': '3 BHK',
                'Bathrooms': '2',
                'Floor': '5th of 12'
            },
            confidence: 92.5
        };
        
        this.displayValuationCard(valuationData);
    }
    
    displayValuationCard(valuationData) {
        const rightSidebarContent = document.getElementById('rightSidebarContent');
        
        rightSidebarContent.innerHTML = `
            <div class="valuation-card">
                <div class="valuation-hero">
                    <div class="valuation-label">Estimated Value</div>
                    <div class="valuation-price">₹${valuationData.price.toFixed(2)} Lakhs</div>
                    <div class="valuation-range">Range: ₹${valuationData.priceRange[0]} - ${valuationData.priceRange[1]} Lakhs</div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Price vs Local Average</div>
                    <canvas id="valuationChart"></canvas>
                </div>
                
                <div class="property-details-list">
                    ${Object.entries(valuationData.features).map(([key, value]) => `
                        <div class="detail-row">
                            <div class="detail-label">
                                <i class="ph ph-${this.getIconForFeature(key)}"></i>
                                <span>${key}</span>
                            </div>
                            <div class="detail-value">${value}</div>
                        </div>
                    `).join('')}
                </div>
                
            </div>
        `;
        
        // Create the chart
        this.createValuationChart(valuationData);
        
        // Show right sidebar
        this.rightSidebar.classList.remove('hidden');
        // Enable side-by-side view
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.classList.add('sidebar-open');
        }
    }
    
    createValuationChart(data) {
        const ctx = document.getElementById('valuationChart');
        if (!ctx) return;
        
        // Destroy existing chart if any
        if (this.currentChart) {
            this.currentChart.destroy();
        }
        
        this.currentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Your Property', 'Local Average'],
                datasets: [{
                    label: 'Price (Lakhs)',
                    data: [data.prediction, data.localAverage],
                    backgroundColor: [
                        'rgba(0, 169, 157, 0.8)',
                        'rgba(136, 136, 136, 0.5)'
                    ],
                    borderColor: [
                        'rgba(0, 169, 157, 1)',
                        'rgba(136, 136, 136, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#2A2A2A',
                        titleColor: '#EAEAEA',
                        bodyColor: '#888888',
                        borderColor: '#333333',
                        borderWidth: 1
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(136, 136, 136, 0.1)'
                        },
                        ticks: {
                            color: '#888888',
                            callback: function(value) {
                                return '₹' + value.toFixed(1) + 'L';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#888888'
                        }
                    }
                }
            }
        });
    }
    
    showDiscoveryView() {
        const rightSidebarContent = document.getElementById('rightSidebarContent');
        
        rightSidebarContent.innerHTML = `
            <div class="discovery-container">
                <h3 style="color: var(--text-primary); margin-bottom: var(--spacing-lg);">Recommended Properties</h3>
                
                ${[1, 2, 3].map(i => `
                    <div class="property-card">
                        <div class="property-image">
                            <div class="property-badge">Featured</div>
                        </div>
                        <div class="property-info">
                            <div class="property-title">Modern ${i + 2} BHK Apartment</div>
                            <div class="property-location">
                                <i class="ph ph-map-pin" style="font-size: 12px;"></i>
                                ${['Whitefield', 'Koramangala', 'Electronic City'][i - 1]}
                            </div>
                            <div class="property-price">₹${75 + i * 25} Lakhs</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        this.rightSidebar.classList.remove('hidden');
    }
    
    showRecommendationsView(recommendations, totalCount) {
        const rightSidebarContent = document.getElementById('rightSidebarContent');
        
        // Generate realistic property images
        const getPropertyImage = (index, prop) => {
            const images = [
                'https://images.unsplash.com/photo-1560184897-ae75f418493e?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80', // Modern living room
                'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80', // Bedroom
                'https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80', // Kitchen
                'https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80', // Modern apartment
                'https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80', // Living room interior
                'https://images.unsplash.com/photo-1567767292278-a4f21aa2d36e?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80', // Modern hall
                'https://images.unsplash.com/photo-1571055107559-3e67626fa8be?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80', // Bedroom interior
                'https://images.unsplash.com/photo-1586105251261-72a756497a11?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80'  // Apartment view
            ];
            return images[index % images.length];
        };
        
        rightSidebarContent.innerHTML = `
            <div class="discovery-container">
                <div class="recommendations-header">
                    <h3 style="color: var(--text-primary); margin-bottom: var(--spacing-xs);">Recommended Properties</h3>
                    <p style="color: var(--text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--spacing-lg);">
                        Showing ${recommendations.length} of ${totalCount} matches
                    </p>
                </div>
                
                ${recommendations.map((prop, index) => `
                    <div class="property-card" style="animation: slideIn 0.3s ease ${index * 0.1}s backwards;">
                        <div class="property-image-container">
                            <img src="${getPropertyImage(index, prop)}" alt="Property Image" class="property-image" 
                                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"/>
                            <div class="property-image-fallback" style="display: none; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                <i class="ph ph-house-line" style="font-size: 48px; color: white; opacity: 0.6;"></i>
                            </div>
                            ${prop.similarity_score ? `
                                <div class="property-badge match-badge">
                                    ${Math.round(prop.similarity_score * 100)}% Match
                                </div>
                            ` : '<div class="property-badge">Featured</div>'}
                        </div>
                        <div class="property-info">
                            <div class="property-title">${prop.bhk || 3} BHK ${prop.property_type || 'Apartment'}</div>
                            <div class="property-location">
                                <i class="ph ph-map-pin"></i>
                                ${prop.location || prop.area_type || 'Prime Location'}
                            </div>
                            <div class="property-details-grid">
                                <div class="detail-item">
                                    <i class="ph ph-ruler"></i>
                                    <span>${prop.sqft || prop.total_sqft || 1500} sqft</span>
                                </div>
                                ${prop.bath ? `
                                    <div class="detail-item">
                                        <i class="ph ph-shower"></i>
                                        <span>${prop.bath} Bath</span>
                                    </div>
                                ` : ''}
                                ${prop.balcony ? `
                                    <div class="detail-item">
                                        <i class="ph ph-sun"></i>
                                        <span>${prop.balcony} Balcony</span>
                                    </div>
                                ` : ''}
                            </div>
                            <div class="property-price-row">
                                <div class="property-price">₹${prop.price.toFixed(1)} L</div>
                                ${prop.price_per_sqft ? `
                                    <div class="price-per-sqft">₹${this.formatNumber(prop.price_per_sqft)}/sqft</div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                `).join('')}
                
                ${totalCount > recommendations.length ? `
                    <div class="more-properties-hint">
                        <i class="ph ph-arrow-down"></i>
                        <span>${totalCount - recommendations.length} more properties available</span>
                    </div>
                ` : ''}
            </div>
        `;
        
        this.rightSidebar.classList.remove('hidden');
        // Enable side-by-side view
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.classList.add('sidebar-open');
        }
    }
    
    showNegotiationView() {
        const rightSidebarContent = document.getElementById('rightSidebarContent');
        
        const negotiations = [
            { party: 'buyer', message: 'Initial offer at ₹175 Lakhs', offer: '₹175L', time: '10:00 AM' },
            { party: 'seller', message: 'Counter at ₹190 Lakhs, considering recent renovations', offer: '₹190L', time: '10:30 AM' },
            { party: 'buyer', message: 'Can go up to ₹180 Lakhs with immediate closure', offer: '₹180L', time: '11:15 AM' },
            { party: 'seller', message: 'Final offer at ₹185 Lakhs', offer: '₹185L', time: '11:45 AM' }
        ];
        
        rightSidebarContent.innerHTML = `
            <div class="negotiation-container">
                <h3 style="color: var(--text-primary); margin-bottom: var(--spacing-lg);">AI Negotiation Timeline</h3>
                
                <div class="negotiation-timeline">
                    ${negotiations.map(item => `
                        <div class="timeline-item">
                            <div class="timeline-dot"></div>
                            <div class="timeline-content">
                                <div class="timeline-header">
                                    <span class="timeline-party ${item.party}">
                                        ${item.party.charAt(0).toUpperCase() + item.party.slice(1)}
                                    </span>
                                    <span class="timeline-time">${item.time}</span>
                                </div>
                                <div class="timeline-message">
                                    ${item.message}
                                    <span class="timeline-offer">${item.offer}</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        this.rightSidebar.classList.remove('hidden');
    }
    
    getIconForFeature(feature) {
        const icons = {
            'Location': 'map-pin',
            'Size': 'ruler',
            'Bedrooms': 'bed',
            'Bathrooms': 'shower',
            'Floor': 'buildings'
        };
        return icons[feature] || 'house';
    }
    
    attachActionButtonListeners(messageDiv) {
        const actionButtons = messageDiv.querySelectorAll('.action-btn');
        
        actionButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handleActionButton(btn.dataset.action, messageDiv);
            });
        });
    }
    
    handleActionButton(action, messageDiv) {
        const messageText = messageDiv.querySelector('.message-text').textContent;
        
        switch(action) {
            case 'copy':
                navigator.clipboard.writeText(messageText).then(() => {
                    const btn = messageDiv.querySelector(`[data-action="copy"]`);
                    btn.classList.add('copied');
                    btn.innerHTML = '<i class="ph ph-check"></i>';
                    setTimeout(() => {
                        btn.classList.remove('copied');
                        btn.innerHTML = '<i class="ph ph-copy"></i>';
                    }, 2000);
                });
                break;
                
            case 'share':
                // Implement share functionality
                console.log('Share feature coming soon');
                break;
                
            case 'explain':
                // Future SHAP values explanation
                console.log('Explanation feature coming soon');
                break;
                
            case 'refine':
                // Pre-fill input with last user message for editing
                const lastUserMessage = this.messageHistory
                    .filter(m => m.sender === 'user')
                    .pop();
                if (lastUserMessage) {
                    this.messageInput.value = lastUserMessage.content;
                    this.messageInput.focus();
                    this.handleInputChange();
                }
                break;
        }
    }

    // Conversation Management
    startNewChat() {
        // Clear messages
        this.chatMessages.innerHTML = '';
        
        // Always show returning user welcome for new chats
        const welcomeContainer = document.createElement('div');
        welcomeContainer.className = 'welcome-container';
        welcomeContainer.id = 'welcomeContainer';
        welcomeContainer.innerHTML = `
            <div class="welcome-simple">
                <p class="welcome-simple-text">How can I help you today?</p>
            </div>
        `;
        
        welcomeContainer.style.display = 'flex';
        this.chatMessages.appendChild(welcomeContainer);
        
        // Reset message history
        this.messageHistory = [];
        
        // Generate new conversation ID
        this.currentConversationId = this.generateId();
        
        // Clear input
        this.messageInput.value = '';
        this.handleInputChange();
        
        // Reset input placeholder
        this.messageInput.placeholder = 'Ask about a property, or select a feature...';
        
        // Hide right sidebar and destroy chart if exists
        this.rightSidebar.classList.add('hidden');
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.classList.remove('sidebar-open');
        }
        if (this.currentChart) {
            this.currentChart.destroy();
            this.currentChart = null;
        }
    }

    loadConversationHistory() {
        // Load sample conversation history
        const sampleConversations = [
            { id: '1', title: '3 BHK in Whitefield', timestamp: '2 hours ago' },
            { id: '2', title: 'Investment Analysis', timestamp: 'Yesterday' },
            { id: '3', title: 'Koramangala Property', timestamp: '3 days ago' }
        ];
        
        sampleConversations.forEach(conv => {
            const convItem = document.createElement('div');
            convItem.className = 'conversation-item';
            convItem.textContent = conv.title;
            convItem.dataset.id = conv.id;
            convItem.addEventListener('click', () => this.loadConversation(conv.id));
            this.conversationHistory.appendChild(convItem);
        });
    }

    loadConversation(id) {
        // Placeholder for loading conversation
        console.log(`Loading conversation ${id}`);
    }

    // Utility Functions
    getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    formatPrice(price) {
        if (!price) return 'N/A';
        
        // Convert to number if string
        const numPrice = typeof price === 'string' ? parseFloat(price) : price;
        
        if (numPrice >= 10000000) {
            // Crores (10 million+)
            return (numPrice / 10000000).toFixed(2) + ' Cr';
        } else if (numPrice >= 100000) {
            // Lakhs (100k+)
            return (numPrice / 100000).toFixed(2) + ' L';
        } else {
            return numPrice.toLocaleString('en-IN');
        }
    }
    
    formatNumber(num) {
        if (!num) return 'N/A';
        const numValue = typeof num === 'string' ? parseFloat(num) : num;
        return numValue.toLocaleString('en-IN', { maximumFractionDigits: 0 });
    }
    
    formatMessage(text) {
        if (!text) return '';
        
        try {
            // Configure marked for better formatting
            if (typeof marked !== 'undefined') {
                marked.setOptions({
                    breaks: true,
                    gfm: true
                });
                
                // Convert markdown to HTML and sanitize
                const html = marked.parse(text);
                return typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(html) : html;
            }
        } catch (error) {
            console.warn('Markdown parsing failed, using plain text formatting:', error);
        }
        
        // Fallback: Basic text formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/\n/g, '<br>') // Line breaks
            .replace(/• /g, '<span style="color: var(--accent-primary);">•</span> ') // Bullet points
            .replace(/(₹[\d,]+(?:\.\d+)?(?:\s*(?:Cr|Crore|L|Lakh|K))?)/g, 
                    '<span style="color: var(--accent-primary); font-weight: 600;">$1</span>'); // Price highlighting
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new RealyticsAI();
    
    // Add some style enhancements for property details
    const style = document.createElement('style');
    style.textContent = `
        .property-details {
            padding: var(--spacing-md);
        }
        
        .property-details h3 {
            font-size: var(--font-size-xl);
            margin-bottom: var(--spacing-lg);
            color: var(--text-primary);
        }
        
        .detail-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: var(--spacing-md);
            margin-bottom: var(--spacing-md);
        }
        
        .detail-card h4 {
            font-size: var(--font-size-base);
            font-weight: 500;
            margin-bottom: var(--spacing-md);
            color: var(--accent-primary);
        }
        
        .detail-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--spacing-sm) 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .detail-item:last-child {
            border-bottom: none;
        }
        
        .detail-item span {
            color: var(--text-secondary);
            font-size: var(--font-size-sm);
        }
        
        .detail-item strong {
            color: var(--text-primary);
            font-weight: 600;
        }
    `;
    document.head.appendChild(style);
    
    console.log('RealyticsAI initialized successfully');
});
