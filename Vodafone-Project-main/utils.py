import geocoder
import cv2
import numpy
from geopy.geocoders import Nominatim

quotes = [
    [
        'Blue light filter or glasses',
        'The blue light that comes from smartphones, tablets, and computers can be a cause for discomfort to the eye if the level of contrast of the screen is not comfortable for viewing. As a workaround, blue filter apps or computer glasses can be used to shield the students’ eyes from this “visual noise.”​'
    ],
    [
        'Alcohol throws your body off balance and leads to poor sleep, especially if you’re dehydrated.',
        ' Even though alcohol may seem to help you fall asleep, you won’t sleep as deeply.'
    ],
    [
        'Stress can zap you of the mental and physical energy needed to carry out your day with ease.',
        ' Stress hormonescan have a negative effect on your sleep patterns, bodily systems, and overall health.'
    ],
    [
        'Proper rest is essential if you want to maintain energy levels throughout the day.',
        ' Relax before going to bed, possibly doing some gentle stretches. Improve your sleep area by keeping it clean and maintaining an appropriate temperature.'
    ],
    [
        'Take regular breaks',
        'Spending long periods of time looking at computer screens is one of the culprits for eye strain. On average, humans blink about 15 to 20 times every minute, but these numbers may decrease when someone is preoccupied, like looking at the computer.'
    ],
    [
        'The benefits of regular exercise are widely recognized.',
        ' Exercise releases endorphins that naturally boosting your energy levels. It can also lead to more high-quality sleep.'
    ],
    [
        'Prepare healthy meals and snacks',
        'Eating healthy and nutritious food and drinks will keep them energetic, focused, and happy throughout the day.'
    ],
    [
        'Lowering your caffeine intake can give you more energy in the long run. ',
        'Though caffeine may give you an initial boost of energy, after it wears off you may be left feeling depleted.'
    ],
    [
        'Drink lots of water',
        'In addition to eating healthy food, children should also develop the habit of drinking adequate amounts of water. For most people, the ideal amount of water to consume is four (4) to six (6) glasses, but this may vary depending on several factors.'
    ],
    [
        'Get a good night’s sleep',
        'Good quality sleep is vital for children’s physical growth, cognitive abilities, and mental health. Experts recommend that children aged 6 to 12 should get 9 to 12 hours of sleep every day, while teenagers need 8 to 10 hours of shuteye to reset their body clock.'
    ],
    [
        'Don’t forget to socialize or have family time',
        'Humans are social beings, and in this period of new normal, students are faced with a different set of challenges. With the lack of personal interaction with their teachers, classmates, or peers, children can easily feel isolated or socially detached. As such, parents and older members of the family should make themselves available and spend after-school quality time as a family.'
    ],
    [
        'Walk',
        'In an ideal world, everybody would visit the gym for strength-training and HIIT exercise on a regular basis. In reality, however, we’re often too busy, tired, or unmotivated to pursue intense exercise. Thankfully, several options exist for integrating physical activity into our daily lives. Walking is easily the simplest and most accessible method.'
    ],
    [
        'Fix Your Posture',
        'Take a moment right now to assess your posture. Are you sitting up straight? Or are you slouched over your computer or smartphone? If you regularly suffer aches and pains, your posture could be a chief culprit. Thankfully, it’s not difficult to fix.'
    ],
    [
        'Fix Your Sleep Environment',
        'Is your living space designed to promote a sound night of sleep? Proper lighting and sound-proofing can make a huge difference, as can reduced exposure to blue light from electronics. Avoid keeping a TV in your bedroom. If possible, use a traditional alarm clock instead of your cell phone. This will reduce the temptation to browse Instagram for hours instead of getting much-needed sleep.'
    ],
    [
        'Make time for fun',
        'Have a favorite activity that helps you unwind? Make time for your hobby. Mental health experts agree that penciling yourself in for some “me time” is an important part of anyone’s life.'
    ],
    [
        'Get out of the house',
        'If you’re able, go outside for some of the items on this list. According to Business Insider, taking a walk in the wilderness or doing a yoga routine in the sunshine can offer surprising health benefits.'
    ]
]
class_labels = ['Speed limit (20km/h)',
                'Speed limit (30km/h)',
                'Speed limit (50km/h)',
                'Speed limit (60km/h)',
                'Speed limit (70km/h)',
                'Speed limit (80km/h)',
                'End of speed limit (80km/h)',
                'Speed limit (100km/h)',
                'Speed limit (120km/h)',
                'No passing',
                'No passing veh over 3.5 tons',
                'Right-of-way at intersection',
                'Priority road',
                'Yield',
                'Stop',
                'No vehicles',
                'Veh > 3.5 tons prohibited',
                'No entry',
                'General caution',
                'Dangerous curve left',
                'Dangerous curve right',
                'Double curve',
                'Bumpy road',
                'Slippery road',
                'Road narrows on the right',
                'Road work',
                'Traffic signals',
                'Pedestrians',
                'Children crossing',
                'Bicycles crossing',
                'Beware of ice/snow',
                'Wild animals crossing',
                'End speed + passing limits',
                'Turn right ahead',
                'Turn left ahead',
                'Ahead only',
                'Go straight or right',
                'Go straight or left',
                'Keep right',
                'Keep left',
                'Roundabout mandatory',
                'End of no passing',
                'End no passing veh > 3.5 tons']


def location():
    g = geocoder.ip('me')
    arr = g.latlng
    try:
        geoLoc = Nominatim(user_agent="GetLoc")
        locname = geoLoc.reverse(f"{arr[0]},{arr[1]}")
        return str(locname.address)
    except Exception:
        return "Not available"


def detectVehicles(frames):
    car_cascade = cv2.CascadeClassifier('data\cars.xml')
    bus_cascade = cv2.CascadeClassifier('data\Bus_front.xml')
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dilated = cv2.dilate(blur, numpy.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    cars = car_cascade.detectMultiScale(closing, 1.1, 3)
    buses = bus_cascade.detectMultiScale(closing, 1.1, 3)
    x1, x2 = 0, 0
    for (x, y, w, h) in cars:
        x1 += x
        x2 += (x + w)
    for (x, y, w, h) in buses:
        x1 += x
        x2 += (x + w)
    # for (x, y, w, h) in cars:
    #     cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x, y, w, h) in buses:
    #     cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return (x1 + x2) // 2


# importing requests and json
import requests, json


def weather_report():
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
    # CITY = "Visakhapatnam"
    API_KEY = "04b2483c337d4418cadef90dadc0e181"
    loc = location()
    # print(loc)
    if loc != "Not available":
        CITY = loc.split(',')[-4][1:]
        # print(CITY)
        URL = BASE_URL + "q=" + CITY + "&appid=" + API_KEY
        response = requests.get(URL)
        if response.status_code == 200:
            data = response.json()
            main = data['main']
            temperature = main['temp'] - 273.15
            humidity = main['humidity']
            pressure = main['pressure']
            report = data['weather']
            # print(f"{CITY:-^30}")
            # print(f"Temperature: {temperature}")
            # print(f"Humidity: {humidity}")
            # print(f"Pressure: {pressure}")
            # print(f"Weather Report: {report[0]['description']}")
            return [int(temperature), humidity, pressure, report[0]['description']]
        else:
            return []
    else:
        return []


# print(weather_report())
