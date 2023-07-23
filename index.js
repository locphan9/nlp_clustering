const PORT = process.env.PORT || 8000
const express = require('express')
const axios = require('axios')
const cheerio = require('cheerio')
const app = express()

const newspapers = [
    {
        name: "theguardian",
        address: "https://www.theguardian.com/technology/artificialintelligenceai",
        base:""
    },
  
    {
        name: "thesun",
        address: "https://www.thesun.co.uk/topic/artificial-intelligence/",
        base:""
    },
    {
        name: "dailymail",
        address: "https://www.dailymail.co.uk/sciencetech/ai/index.html",
        base:""
    },
    {
        name: "nyp",
        address: "https://nypost.com/tag/artificial-intelligence/",
        base:""
    }
]    

const articles = []

newspapers.forEach(newspaper => {
    axios.get(newspaper.address)
        .then(response => {
            const html = response.data
            const $ = cheerio.load(html)

            $('a:contains("AI")', html).each(function () {
                const title = $(this).text()
                const url = $(this).attr('href')
                articles.push({
                    title,
                    url: newspaper.base + url,
                    source: newspaper.name
                })
            })

        })
})


app.get('/', (req, res) => {
    res.json('Welcome to my Climate Change News API')
})

app.get('/news', (req, res) => {
    res.json(articles)
})

app.get('/news/:newspaperId', (req, res) => {
    const newspaperId = req.params.newspaperId

    const newspaperAddress = newspapers.filter(newspaper => newspaper.name == newspaperId)[0].address
    const newspaperBase = newspapers.filter(newspaper => newspaper.name == newspaperId)[0].base


    axios.get(newspaperAddress)
        .then(response => {
            const html = response.data
            const $ = cheerio.load(html)
            const specificArticles = []

            $('a:contains("AI")', html).each(function () {
                const title = $(this).text()
                const url = $(this).attr('href')
                specificArticles.push({
                    title,
                    url: newspaperBase + url,
                    source: newspaperId
                })
            })
            res.json(specificArticles)
        }).catch(err => console.log(err))
        console.log(specificArticles)

})
console.log(articles)
app.listen(PORT, () => console.log(`server running on PORT ${PORT}`))