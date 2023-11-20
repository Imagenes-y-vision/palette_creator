'use client'

import Image from 'next/image'
import styles from './styles.module.css'
import { useEffect, useState } from 'react'
import Link from 'next/link'


type ImagesResponse = {
  page: number;
  results: number;
  total: number;
  total_pages: number;
  images: {
      id: number;
      title: string;
      url: string;
      description: string;
  }[];
}

export default function Home(params: { params: {}, searchParams: {filter?: string}}) {
  const initialData = {
    page: 0,
    results: 0,
    total: 0,
    total_pages: 0,
    images: []
  }
  // page state
  const [page, setPage] = useState(1)
  const [data, setData] = useState<ImagesResponse>(initialData)
  const [filter, setFilter] = useState<string[]>()

  console.log('params', params.searchParams.filter)

  useEffect(() => {
    fetchImages()
  }, [page])

  useEffect(() => {
    if (params.searchParams.filter)
      setFilter(params.searchParams.filter.split(','))
  }, [params.searchParams.filter])

  // fetch data from API
  const fetchImages = async () => {
    if (filter) {
      const response = await fetch(`http://localhost:8000?page=${page}&limit=6&filter=${filter.join(',')}`)
      const data_ = await response.json()
      setData(data_)
      return
    }
    try {
    const response = await fetch(`http://localhost:8000?page=${page}&limit=6`);
     const data_ = await response.json();
    setData(data_);
    } catch (error) {
      console.error("Error al realizar la solicitud fetch:", error);
      }
  }

  // page change handler
  const handlePageChange = (page: number) => {
    setPage(page)
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className={styles.title}>Galería de imágenes Beatriz Gonzáles</h1>

      {/* Page number info */}
      <div className={`flex justify-between items-center w-full ${styles.page_info}`}>
        <p className="text-gray-500">Página {page} de {data.total_pages || 1}</p>
        <p className="text-gray-500">Mostrando {data.results} resultados</p>
      </div>

      {/* Show the color filter applied */}
      {filter && (
        <div className={`flex flex-row items-center ${styles.filter}`}>
          <p className="text-gray-500">Filtro aplicado:</p>
          <div className={`w-12 h-12 m-1 ${styles.palette_color}`}
            style={{ backgroundColor: `rgb(${filter[0]}, ${filter[1]}, ${filter[2]})` }}></div>

          {/* Button to remove the filter */}
          <Link href="/">
            <button className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded"
              onClick={() => {setFilter(undefined); setPage(1)}}
            >
              Remover filtro
            </button>
          </Link>
        </div>
      )}

      {/* Images list */}
      <div className={styles.images_list}>
        {data.images?.map((image) => (
          <div key={image.id} className="flex flex-col items-center justify-center">
            <div>
            <Link href={`/image/${image.id}`}>
              <Image
                src={image.url}
                width={300}
                height={300}
                alt={image.title}
                className={styles.image}
              />
            </Link>
            <p className="text-center">{image.title}</p>
            </div>
            <div>
              <p>{image.description}</p>
            </div>
          </div>))
      }
        </div>

        {/* Paginator */}
        <div className="flex justify-between items-center">
          <button className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-l"
          disabled={page === 1}
          onClick={() => handlePageChange(page - 1)}
          >
            Anterior
          </button>
          <button className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-r"
          disabled={page === data.total_pages}
          onClick={() => handlePageChange(page + 1)}
          >
            Siguiente
          </button>
        </div>
    </main>
  )
}
