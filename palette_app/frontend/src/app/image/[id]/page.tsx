'use client'

import Image from 'next/image'
import styles from './styles.module.css'
import { useEffect, useState } from 'react'
import Link from 'next/link'

type ImageResponse = {
  id: number;
  title: string;
  url: string;
  description: string;
  palette: number[][];
}

export default function ImageDetail({ params }: { params: { id: string }}) {
  const [data, setData] = useState<ImageResponse>({} as ImageResponse)
  const [availableMethods, setAvailableMethods] = useState<string[]>([])
  const [selectedMethod, setSelectedMethod] = useState<string>('')

  useEffect(() => {
    fetchMethods()
    loadSelectedMethod()
  }, [])

  useEffect(() => {
    fetchImage()
  }, [selectedMethod])

  // fetch data from API
  const fetchImage = async () => {
    let response
    if (selectedMethod) {
      response = await fetch(`http://localhost:8000/image/${params.id}?method=${selectedMethod}`)
    } else {
      response = await fetch(`http://localhost:8000/image/${params.id}`)
    }
    const data_ = await response.json()
    setData(data_)
  }

  const fetchMethods = async () => {
    const response = await fetch(`http://localhost:8000/methods`)
    const data_ = await response.json()
    setAvailableMethods(data_)
  }

  const loadSelectedMethod = () => {
    const selectedMethod = localStorage.getItem('selectedMethod')
    setSelectedMethod(selectedMethod || '')
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className={styles.title}>Galería de imágenes Beatriz Gonzáles</h1>

      {/* Page number info */}
      <div className={`flex justify-between flex-col w-full ${styles.page_info}`}>
        <p className="text-gray-500">Título de la imagen: {data.title}</p>
        <p className="text-gray-500">{data.description}</p>
      </div>

      <Image
        src={data.url}
        width={300}
        height={300}
        alt={data.title}
        className={styles.image}
      />

      {/* Select with available methods */}
      <select className="w-1/2 h-10 m-4 p-2 border border-gray-400 rounded-md"
        value={selectedMethod}
        onChange={(e) => setSelectedMethod(e.target.value)}
      >
        {availableMethods.map((method, index) => (
          <option key={index}>{method}</option>
        ))}
      </select>

      {/* Show the colors palette of the image */}
      <div className={`flex flex-row flex-wrap justify-center ${styles.palette}`}>
        {data.palette?.map((color, index) => (
          <Link href={`/?filter=${color[0]},${color[1]},${color[2]}`}>
          <div
            key={index}
            className={`w-12 h-12 m-1 ${styles.palette_color}`}
            style={{ backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})` }}
          />
          </Link>
        ))}
      </div>
    </main>
  )
}
