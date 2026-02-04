




class BidRequest:
    serialVersionUID = -3012027079030559912
    def __init__(self):
        self.bidId = None
        self.timestamp = None
        self.visitorId = None
        self.userAgent = None
        self.ipAddress = None
        self.region = None
        self.city = None
        self.adExchange = None
        self.domain = None
        self.url = None
        self.anonymousURLID = None
        self.adSlotID = None
        self.adSlotWidth = None
        self.adSlotHeight = None
        self.adSlotVisibility = None
        self.adSlotFormat = None
        self.adSlotFloorPrice = None
        self.creativeID = None
        self.advertiserId = None
        self.userTags = None

    def getBidId(self) -> str:
        return self.bidId
    
    
    def setBidId(self, bidId: str):
        self.bidId = bidId

    
    def getTimestamp(self) -> str:
        return self.timestamp
    
    
    def setTimestamp(self, timestamp: str):
        self.timestamp = timestamp

    
    def getVisitorId(self) -> str:
        return self.visitorId
    
    
    def setVisitorId(self, visitorId: str):
        self.visitorId = visitorId

    
    def getUserAgent(self) -> str:
        return self.userAgent
    

    def setUserAgent(self, userAgent: str):
        self.userAgent = userAgent


    def getIpAddress(self) -> str:
        return self.ipAddress
    

    def setIpAddress(self, ipAddress: str):
        self.ipAddress = ipAddress


    def getRegion(self) -> str:
        return self.region
    

    def setRegion(self, region: str):
        self.region = region

    
    def getCity(self) -> str:
        return self.city    
    

    def setCity(self, city: str):
        self.city = city


    def getAdExchange(self) -> str:    
        return self.adExchange
    

    def setAdExchange(self, adExchange: str):  
        self.adExchange = adExchange

    
    def getDomain(self) -> str:    
        return self.domain
    

    def setDomain(self, domain: str):    
        self.domain = domain


    def getUrl(self) -> str:    
        return self.url
    

    def setUrl(self, url: str):    
        self.url = url


    def getAnonymousURLID(self) -> str:    
        return self.anonymousURLID
    

    def setAnonymousURLID(self, anonymousURLID: str):    
        self.anonymousURLID = anonymousURLID


    def getAdSlotID(self) -> str:    
        return self.adSlotID
    

    def setAdSlotID(self, adSlotID: str):    
        self.adSlotID = adSlotID


    def getAdSlotWidth(self) -> str:    
        return self.adSlotWidth
    

    def setAdSlotWidth(self, adSlotWidth: str):   
        self.adSlotWidth = adSlotWidth


    def getAdSlotHeight(self) -> str:   
        return self.adSlotHeight
    

    def setAdSlotHeight(self, adSlotHeight: str):
        self.adSlotHeight = adSlotHeight


    def getAdSlotVisibility(self) -> str:
        return self.adSlotVisibility
    

    def setAdSlotVisibility(self, adSlotVisibility: str):
        self.adSlotVisibility = adSlotVisibility


    def getAdSlotFormat(self) -> str:
        return self.adSlotFormat
    

    def setAdSlotFormat(self, adSlotFormat: str):
        self.adSlotFormat = adSlotFormat


    def getAdSlotFloorPrice(self) -> str:
        return self.adSlotFloorPrice
    

    def setAdSlotFloorPrice(self, adSlotFloorPrice: str):
        self.adSlotFloorPrice = adSlotFloorPrice


    def getCreativeID(self) -> str:
        return self.creativeID
    

    def setCreativeID(self, creativeID: str):
        self.creativeID = creativeID


    def getAdvertiserId(self) -> str:   
        return self.advertiserId 
    

    def setAdvertiserId(self, advertiserId: str):
        self.advertiserId = advertiserId


    def getUserTags(self) -> str:
        return self.userTags
    

    def setUserTags(self, userTags: str):
        self.userTags = userTags


    